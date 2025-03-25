import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs
from experiments.evaluation.compute_metrics import aggregate_tcav_metrics, compute_tcav_metrics_batch
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import MODELS_1D, get_fn_model_loader
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
    
    measure_quality_cav_gt(config)

def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

def measure_quality_cav_gt(config):
    """ Computes TCAV scores w.r.t. ground truth CAV in controlled settings
    Args:
        config (dict): config for model correction run
    """

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name']
    model_name = config['model_name']
    cav_mode = config.get("cav_mode", "cavs_max")
    data_paths = config['data_paths']
    artifact_name = config["artifact"]
    img_size = config.get('img_size', 224)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    config["device"] = device

    if "funnybirds_ch" in dataset_name:
        data_paths = [f"{data_paths[0]}/train", f"{data_paths[0]}/test_ch"]
    artifacts_file = config.get('artifacts_file', None)

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=config.get('p_artifact', 1.0),
                                        artifact_ids_file=artifacts_file,
                                        image_size=img_size,
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        **artifact_kwargs, **dataset_specific_kwargs)
    
    dataset_clean = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=0,
                                        artifact_ids_file=artifacts_file,
                                        image_size=img_size,
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        **artifact_kwargs, **dataset_specific_kwargs)
    
    if "funnybirds_ch" in dataset_name:
        base_path ="/".join(data_paths[-1].split("/")[:-1])
        data_paths[-1] = f"{base_path}/test_clean"

    n_classes = len(dataset.class_names)
    config_name = os.path.basename(config['config_file'])[:-5]
    
    ckpt_path = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"

    if config["num_epochs"] == 0 and dataset_name == "imagenet":
        ckpt_path = None

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device)
    model = prepare_model_for_evaluation(model, dataset, ckpt_path, device, config)
    
    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

    results = {}
    for split in [
        'train',
        'test',
        'val'
    ]:
        split_set = sets[split]
        artifact_ids_split = [i for i in dataset.sample_ids_by_artifact[artifact_name] if i in split_set]
        dataset_artifact_only = dataset.get_subset_by_idxs(artifact_ids_split)
        dataset_clean_artifact_only = dataset_clean.get_subset_by_idxs(artifact_ids_split)
        dl_art = DataLoader(dataset_artifact_only, batch_size=1, shuffle=False)
        dl_clean = DataLoader(dataset_clean_artifact_only, batch_size=1, shuffle=False)

        
        # Register forward hook for layer of interest
        layer = config["layer_name"]
        for n, m in model.named_modules():
            if n.endswith(layer):
                m.register_forward_hook(get_activation)

        if dataset_name in ("isic", "imagenet"):
            attacked_class = None
        else:
            # controlled setting
            attacked_class = dataset.get_class_id_by_name(dataset.attacked_classes[0])
        

        TCAV_sens_list = []
        TCAV_pos = 0
        TCAV_neg = 0
        TCAV_pos_mean = 0
        TCAV_neg_mean = 0
        for batch_art, batch_clean in tqdm.tqdm(zip(dl_art, dl_clean)):
            x_att, y = batch_art
            x_clean, _ = batch_clean

            model(x_clean.to(device))
            x_clean_latent = activations.clone()
            model.zero_grad()

            grad_target = attacked_class if attacked_class is not None else y

            # Compute latent representation
            with torch.enable_grad():
                x_att.requires_grad = True
                x_att = x_att.to(device)
                y_hat = model(x_att)
                yc_hat = y_hat[:, grad_target]

                grad = torch.autograd.grad(outputs=yc_hat,
                                           inputs=activations,
                                           retain_graph=True,
                                           grad_outputs=torch.ones_like(yc_hat))[0]

                grad = grad.detach().cpu()
                model.zero_grad()
                
                x_att_latent = activations.clone()
                cav_sample = (x_att_latent - x_clean_latent).detach().cpu()
                if (cav_mode == "cavs_max") and not (any([n in model_name for n in MODELS_1D])):
                    cav_sample = cav_sample.flatten(start_dim=2).max(2).values

                tcav_metrics_batch = compute_tcav_metrics_batch(grad, cav_sample)
                        
                TCAV_pos += tcav_metrics_batch["TCAV_pos"]
                TCAV_neg += tcav_metrics_batch["TCAV_neg"]
                TCAV_pos_mean += tcav_metrics_batch["TCAV_pos_mean"]
                TCAV_neg_mean += tcav_metrics_batch["TCAV_neg_mean"]

                TCAV_sens_list.append(tcav_metrics_batch["TCAV_sensitivity"])

        tcav_metrics = aggregate_tcav_metrics(TCAV_pos, TCAV_neg, TCAV_pos_mean, TCAV_neg_mean, TCAV_sens_list)

        results[f"{split}_mean_tcav_quotient_gt"] = tcav_metrics['mean_tcav_quotient']
        results[f"{split}_mean_quotient_gt_sderr"] = tcav_metrics['mean_quotient_sderr']
        results[f"{split}_tcav_quotient_gt"] = tcav_metrics['tcav_quotient']
        results[f"{split}_quotient_gt_sderr"] = tcav_metrics['quotient_sderr']
        results[f"{split}_mean_tcav_sensitivity_gt"] = tcav_metrics['mean_tcav_sensitivity']
        results[f"{split}_mean_tcav_sensitivity_gt_sem"] = tcav_metrics['mean_tcav_sensitivity_sem']

        if config.get('wandb_api_key', None):
            wandb.log({**results, **config})


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
