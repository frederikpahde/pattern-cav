import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
import wandb
import yaml
from crp.attribution import CondAttribution
from crp.helper import get_layer_names
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset, get_dataset_kwargs
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_canonizer, get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/clarc/isic_attacked/local/vgg16_RRClarc_lamb100000_signal_cavs_max_sgd_lr0.0005_features.28.yaml")

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['config_file'] = args.config_file

    compute_artifact_relevance(config)


def compute_artifact_relevance(config):
    """
    Computes average relevance in artifactual regions for train/val/test splits.

    Args:
        config (dict): experiment config
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name'] + "_hm"
    model_name = config['model_name']
    artifacts_file = config.get('artifacts_file', None)
    config_name = os.path.basename(config["config_file"])[:-5]

    kwargs_data = {
        "p_artifact": 1.0,
        "artifact_type": config['artifact_type'],
        "attacked_classes": config['attacked_classes']
    } if config['artifact'] == 'artificial' else {}

    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)

    data_paths=config['data_paths']
        
    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        artifact_ids_file=artifacts_file,
                                        artifact=config['artifact'],
                                        **kwargs_data, **artifact_kwargs, **dataset_specific_kwargs)

    ckpt_path = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    model = get_fn_model_loader(model_name=model_name)(n_class=dataset.class_names.__len__(), ckpt_path=ckpt_path)
    model = prepare_model_for_evaluation(model, dataset, ckpt_path, device, config)

    attribution = CondAttribution(model)
    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)

    layer_names = get_layer_names(model, [torch.nn.Conv2d])

    artifact_labels = list(dataset.sample_ids_by_artifact.keys())
    artifact_sample_ids = list(dataset.sample_ids_by_artifact.values())
    scores = []
    sems = []

    train_set = dataset.idxs_train
    test_set = dataset.idxs_test
    val_set = dataset.idxs_val

    splits = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

    gaussian = torchvision.transforms.GaussianBlur(kernel_size=41, sigma=8.0)

    for split in ['train', 'val', 'test']:

        split_set = splits[split]
        sample_sets_split = [[y for y in x if y in split_set] for x in artifact_sample_ids]

        for k, samples in enumerate(sample_sets_split):

            dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                                normalize_data=True,
                                                artifact_ids_file=artifacts_file,
                                                artifact=artifact_labels[k],
                                                **kwargs_data, **artifact_kwargs, **dataset_specific_kwargs)

            n_samples = len(samples)
            n_batches = int(np.ceil(n_samples / config['batch_size']))
            score = []
            for i in tqdm(range(n_batches)):
                samples_batch = samples[i * config['batch_size']:(i + 1) * config['batch_size']]
                data = torch.stack([dataset[j][0] for j in samples_batch], dim=0).to(device).requires_grad_()
                targets = torch.stack([dataset[j][1] for j in samples_batch], dim=0).to(device)

                condition = [{"y": c_id} for c_id in targets]

                attr = attribution(data, condition, composite, record_layer=layer_names, init_rel=1)

                # load mask as third entry from data sample
                mask = torch.stack([dataset[j][2] for j in samples_batch], dim=0).to(device)
                mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.1)
                mask = gaussian(mask.clamp(min=0)) ** 1.0
                mask = 1.0 * (mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None] > 0.3)

                inside = (attr.heatmap * mask).abs().sum((1, 2)) / (
                        attr.heatmap.abs().sum((1, 2)) + 1e-10)
                score.extend(list(inside.detach().cpu()))

            scores.append(np.mean(score))
            sems.append(np.std(score) / np.sqrt(len(score)))
            
            if config.get('wandb_api_key', None):
                wandb.log({f"{split}_artifact_rel_{artifact_labels[k].lower()}": scores[-1]})
                wandb.log({f"{split}_artifact_rel_{artifact_labels[k].lower()}_sem": sems[-1]})


if __name__ == "__main__":
    main()
