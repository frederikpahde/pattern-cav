import logging
import os
from argparse import ArgumentParser
from distutils.util import strtobool

import numpy as np
import torch
import yaml
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset, get_dataset_kwargs
from models import MODELS_1D, get_canonizer, get_fn_model_loader, TRANSFORMER_MODELS
from utils.artificial_artifact import get_artifact_kwargs

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/clarc/isic_attacked/local/vgg16_RRClarc_lamb100000_signal_cavs_max_sgd_lr0.0005_features.28.yaml")
    parser.add_argument("--class_id", default=0, type=int)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--split", default="all", choices=['train', 'val', 'all'], type=str)
    args = parser.parse_args()

    return args


def str2bool(s):
    if isinstance(s, str):
        return strtobool(s)
    return bool(s)


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)

    config['config_file'] = args.config_file
    config['split'] = args.split
    config['class_id'] = args.class_id

    run_collect_relevances_and_activations(config)


def run_collect_relevances_and_activations(config):
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name']
    data_paths = config['data_paths']
    img_size = config.get("img_size", 224)
    attacked_classes = config.get("attacked_classes", [])
    p_artifact = config.get("p_artifact", None)
    # p_artifact = 1
    artifact_type = config.get("artifact_type", None)
    split = config["split"]
    model_name = config['model_name']

    ckpt_path = config["ckpt_path"]
    batch_size = config['batch_size']
    results_dir = config.get('dir_precomputed_data', 'results')
    class_name = config['class_name']

    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        image_size=img_size,
                                        attacked_classes=attacked_classes,
                                        p_artifact=p_artifact,
                                        artifact_type=artifact_type,
                                        **artifact_kwargs, **dataset_specific_kwargs)

    if split != "all":
        if split == 'train':
            dataset_split = dataset.get_subset_by_idxs(dataset.idxs_train)
        elif split == 'val':
            dataset_split = dataset.get_subset_by_idxs(dataset.idxs_val)
        elif split == 'test':
            dataset_split = dataset.get_subset_by_idxs(dataset.idxs_test)
        else:
            raise ValueError(f"Unknown split {split}")
    else:
        dataset_split = dataset

    logger.info(f"Using split {split} ({len(dataset_split)} samples)")

    n_classes = len(dataset_split.class_names)

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path)
    model = model.to(device)
    model.eval()

    attribution = CondAttribution(model)
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    cc = ChannelConcept()

    samples = np.array(
        [i for i in range(len(dataset_split)) if ((class_name is None) or (dataset_split.get_target(i) == class_name))])
    logger.info(f"Found {len(samples)} samples of class {class_name}.")

    n_samples = len(samples)
    n_batches = int(np.ceil(n_samples / batch_size))

    if ("resnet" in model_name) or ("efficientnet" in model_name):
        layer_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Identity)]
    else:
        layer_names = [n for n, m in model.named_modules() if isinstance(m, (torch.nn.Identity, torch.nn.Conv2d))]

    if any([m in model_name for m in TRANSFORMER_MODELS]):
        layer_names = [l for l in layer_names if ("inspection" in l)]

    ## quick hack
    layer_names = [config["layer_name"]]

    #layer_names = [layer_name]
    crvs = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_max = dict(zip(layer_names, [[] for _ in layer_names]))
    cavs_mean = dict(zip(layer_names, [[] for _ in layer_names]))
    smpls = []
    output = []

    for i in tqdm(range(n_batches)):
        samples_batch = samples[i * batch_size:(i + 1) * batch_size]
        data = torch.stack([dataset_split[j][0] for j in samples_batch], dim=0).to(device).requires_grad_()
        out = model(data).detach().cpu()
        condition = [{"y": c_id} for c_id in out.argmax(1)]
        attr = attribution(data, condition, composite, record_layer=layer_names, init_rel=1)
        non_zero = ((attr.heatmap.sum((1, 2)).abs().detach().cpu() > 0) * (out.argmax(1) == dataset.get_class_id_by_name(class_name))).numpy()
        non_zero = np.array([True for i in non_zero])
        samples_nz = samples_batch[non_zero]
        output.append(out[non_zero])

        layer_names_ = [l for l in layer_names if l in attr.relevances.keys()]

        if samples_nz.size:
            if any([n in model_name for n in MODELS_1D]):
                smpls += [s for s in samples_batch]
                acts_max = [attr.activations[layer] for layer in layer_names_]
                acts_mean = [attr.activations[layer] for layer in layer_names_]
                
                for l, amax, amean in zip(layer_names_, acts_max, acts_mean):
                    cavs_max[l] += amax.detach().cpu()
                    cavs_mean[l] += amean.detach().cpu()
            else:
                lnames = [lname for lname, acts in attr.activations.items() if acts.dim() == 4]
                smpls += [s for s in samples_nz]
                rels = [cc.attribute(attr.relevances[layer][non_zero], abs_norm=True) for layer in lnames]
                if "swin_former" in model_name:
                    # swinformer has activations in form BxHxWxC
                    acts_max = [attr.activations[layer][non_zero].transpose(1,3).transpose(2,3).flatten(start_dim=2).max(2)[0] for layer in lnames]
                    acts_mean = [attr.activations[layer][non_zero].transpose(1,3).transpose(2,3).mean((2, 3)) for layer in lnames]
                else:
                    acts_max = [attr.activations[layer][non_zero].flatten(start_dim=2).max(2)[0] for layer in lnames]
                    acts_mean = [attr.activations[layer][non_zero].mean((2, 3)) for layer in lnames]
                for l, r, amax, amean in zip(lnames, rels, acts_max, acts_mean):
                    crvs[l] += r.detach().cpu()
                    cavs_max[l] += amax.detach().cpu()
                    cavs_mean[l] += amean.detach().cpu()
    
    if dataset_name == "isic":
        artifact_extension = ""
    else:
        artifact_extension = f"_{artifact_type}-{p_artifact}" if p_artifact is not None else ""

    path = f"{results_dir}/global_relevances_and_activations/{dataset_name}{artifact_extension}/{model_name}"
    os.makedirs(path, exist_ok=True)

    print("saving as", f"{path}/class_{class_name}_{split}.pth")

    str_class_id = 'all' if class_name is None else class_name
    torch.save({"samples": smpls,
                "output": output,
                "crvs": crvs,
                "cavs_max": cavs_max,
                "cavs_mean": cavs_mean},

               f"{path}/class_{str_class_id}_{split}.pth")
    for layer in layer_names:
        if layer in crvs.keys():
            torch.save({"samples": smpls,
                        "output": output,
                        "crvs": crvs[layer],
                        "cavs_max": cavs_max[layer],
                        "cavs_mean": cavs_mean[layer]},
                    f"{path}/{layer}_class_{str_class_id}_{split}.pth")


if __name__ == "__main__":
    main()
