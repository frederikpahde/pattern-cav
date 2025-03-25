import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision
import wandb
import yaml
from tqdm import tqdm

from datasets import get_dataset
from experiments.evaluation.compute_metrics import aggregate_tcav_metrics, compute_tcav_metrics_batch
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--artifact", default="band_aid")
    parser.add_argument('--config_file',
                        default="config_files/real_artifacts_clarc/isic/local/vgg16_band_aid_RRClarc_lamb100000_signal_cavs_max_sgd_lr0.0005_features.28.yaml")

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
    config['attack_artifact'] = args.artifact

    if args.artifact in args.config_file:
        measure_tcav_gt_auto_attacked(config)
    else:
        print(f"Skipping eval wrt {args.artifact} for {args.config_file}")
    

def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

def measure_tcav_gt_auto_attacked(config):
    """
    Computes TCAV score w.r.t. ground truth CAV for auto-attacked samples (artifact copy/pasted onto clean samples)

    Args:
        config (dict): exeriment config
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name']
    model_name = config['model_name']
    config_name = os.path.basename(config["config_file"])[:-5]
    
    dataset = get_dataset(f"{dataset_name}_hm")(data_paths=config['data_paths'],
                                                normalize_data=True,
                                                artifact_ids_file=config['artifacts_file'],
                                                artifact=config['attack_artifact'])

    n_classes = len(dataset.class_names)
    cav_mode = config.get("cav_mode", "cavs_max")
    ckpt_path = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"

    rng = np.random.default_rng(0)

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path)
    model = prepare_model_for_evaluation(model, dataset, ckpt_path, device, config)

    # Register forward hook for layer of interest
    layer = config["layer_name"]
    for n, m in model.named_modules():
        if n.endswith(layer):
            m.register_forward_hook(get_activation)

    gaussian = torchvision.transforms.GaussianBlur(kernel_size=41, sigma=5.0)

    ### COLLECT ARTIFACTS
    artifact_samples = dataset.sample_ids_by_artifact[config['attack_artifact']]
    masks = []
    artifacts = []
    labels = []
    batch_size = config['batch_size']
    print(f"There are {len(artifact_samples)} artifact samples")
    for k, samples in enumerate([artifact_samples]):

        n_samples = len(samples)
        n_batches = int(np.ceil(n_samples / batch_size))

        for i in tqdm(range(n_batches)):
            samples_batch = samples[i * batch_size:(i + 1) * batch_size]
            data = torch.stack([dataset[j][0] for j in samples_batch], dim=0)
            labels_batch = torch.stack([dataset[j][1] for j in samples_batch], dim=0)
            mask = torch.stack([dataset[j][2] for j in samples_batch])
            mask = gaussian(mask.clamp(min=0)) ** 1.0
            mask = mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None]
            artifacts.append(data)
            masks.append(mask)
            labels.append(labels_batch)

    artifact_sample_labels = torch.cat(labels)
    masks = torch.cat(masks, 0)
    artifacts = torch.cat(artifacts, 0)

    val_set = dataset.idxs_val
    test_set = dataset.idxs_test

    sets = {
        "val": val_set,
        "test": test_set
    }

    metrics = {}

    for split in ['val', 'test']:
        split_set = sets[split]
        sample_sets = [split_set,
                       [x for x in split_set if (x not in artifact_samples)]]

        print("size of sample sets", [len(x) for x in sample_sets])

        for k, samples in enumerate(sample_sets):

            
            samples = np.array(samples)
            n_samples = len(samples)
            n_batches = int(np.ceil(n_samples / batch_size))

            TCAV_sens_list = []
            TCAV_pos = 0
            TCAV_neg = 0
            TCAV_pos_mean = 0
            TCAV_neg_mean = 0

            for _ in range(1):
                for i in tqdm(range(n_batches)):
                    samples_batch = samples[i * batch_size:(i + 1) * batch_size]
                    data = torch.stack([dataset[j][0] for j in samples_batch], dim=0)
                    y = torch.stack([dataset[j][1] for j in samples_batch], dim=0)
                    pick = rng.choice(range(len(masks)), len(samples_batch))
                    m = masks[pick][:, None, :, :]
                    artifact = artifacts[pick]
                    artifact_sample_labels_picked = artifact_sample_labels[pick]

                    x_clean = data.clone()
                    x_att = data * (1 - m) + artifact * m

                    model(x_clean.to(device))
                    x_clean_latent = activations.clone()

                    # grad_target = attacked_class if attacked_class is not None else y
                    grad_target = artifact_sample_labels_picked

                    # Compute latent representation
                    with torch.enable_grad():
                        x_att.requires_grad = True
                        x_att = x_att.to(device)
                        y_hat = model(x_att)
                        yc_hat = torch.gather(y_hat, 1, grad_target.view(-1, 1).to(device))

                        grad = torch.autograd.grad(outputs=yc_hat,
                                                inputs=activations,
                                                retain_graph=True,
                                                grad_outputs=torch.ones_like(yc_hat))[0]

                        grad = grad.detach().cpu()
                        model.zero_grad()
                        
                        x_att_latent = activations.clone()
                        cav_sample = (x_att_latent - x_clean_latent).detach().cpu()

                        if cav_mode == "cavs_max":
                            cav_sample = cav_sample if cav_sample.dim() > 2 else cav_sample[..., None, None]
                            cav_sample = cav_sample.flatten(start_dim=2).max(2).values

                        tcav_metrics_batch = compute_tcav_metrics_batch(grad, cav_sample)
                        
                        TCAV_pos += tcav_metrics_batch["TCAV_pos"]
                        TCAV_neg += tcav_metrics_batch["TCAV_neg"]
                        TCAV_pos_mean += tcav_metrics_batch["TCAV_pos_mean"]
                        TCAV_neg_mean += tcav_metrics_batch["TCAV_neg_mean"]

                        TCAV_sens_list.append(tcav_metrics_batch["TCAV_sensitivity"])
            
            tcav_metrics = aggregate_tcav_metrics(TCAV_pos, TCAV_neg, TCAV_pos_mean, TCAV_neg_mean, TCAV_sens_list)

            metrics[f"{split}_mean_tcav_quotient_gt_auto-attacked_{config['attack_artifact']}"] = tcav_metrics['mean_tcav_quotient']
            metrics[f"{split}_mean_quotient_gt_sderr_auto-attacked_{config['attack_artifact']}"] = tcav_metrics['mean_quotient_sderr']

            metrics[f"{split}_tcav_quotient_gt_auto-attacked_{config['attack_artifact']}"] = tcav_metrics['tcav_quotient']
            metrics[f"{split}_quotient_gt_sderr_auto-attacked_{config['attack_artifact']}"] = tcav_metrics['quotient_sderr']

            metrics[f"{split}_mean_tcav_sensitivity_gt_auto-attacked_{config['attack_artifact']}"] = tcav_metrics['mean_tcav_sensitivity']
            metrics[f"{split}_mean_tcav_sensitivity_gt_sem_auto-attacked_{config['attack_artifact']}"] = tcav_metrics['mean_tcav_sensitivity_sem']

            if config.get('wandb_api_key', None):
                wandb.log(metrics)

if __name__ == "__main__":
    main()
