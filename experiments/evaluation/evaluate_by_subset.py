import os
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs
from experiments.evaluation.compute_metrics import compute_metrics, compute_model_scores
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", default=None)
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

    config['ckpt_path'] = args.ckpt_path
    config['config_file'] = args.config_file


    evaluate_by_subset(config)

def evaluate_by_subset(config):
    """Run evaluations for all data splits and sets of artifacts

    Args:
        config (dict): model correction run config
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = config['dataset_name']
    model_name = config['model_name']
    config_name = os.path.basename(config["config_file"])[:-5]

    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    limit_train_batches = config.get("limit_train_batches", None)
    dataset = get_dataset(dataset_name)(data_paths=config['data_paths'],
                                        normalize_data=True,
                                        artifact_ids_file=config['artifacts_file'],
                                         **artifact_kwargs, **dataset_specific_kwargs)

    n_classes = len(dataset.class_names)
    ckpt_path = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    
    if config["num_epochs"] == 0 and dataset_name == "imagenet":
        ckpt_path = None

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path)
    model = prepare_model_for_evaluation(model, dataset, ckpt_path, device, config)

    sets = {
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
    }

    labels_set = list(dataset.sample_ids_by_artifact.keys())
    sample_sets = list(dataset.sample_ids_by_artifact.values())

    clean_idxs = np.arange(len(dataset))
    for k, v in dataset.sample_ids_by_artifact.items():
        clean_idxs = np.setdiff1d(clean_idxs, v)
    sample_sets.append(clean_idxs)
    labels_set.append("clean")

    for split in ['test', 'val']:
        split_set = sets[split]
        sample_sets_split = [[y for y in x if y in split_set] for x in sample_sets]

        model_outs_all = []
        ys_all = []
        print(f"size of sample sets ({split})", [len(x) for x in sample_sets_split])

        for k, samples in enumerate(sample_sets_split):
            if len(samples) > 0:
                samples = np.array(samples)
                dataset_subset = dataset.get_subset_by_idxs(samples)
                dl_subset = DataLoader(dataset_subset, batch_size=config['batch_size'], shuffle=False)
                model_outs, y_true = compute_model_scores(model, dl_subset, device, limit_train_batches)
                class_names = None if "imagenet" in dataset_name else dataset.class_names
                metrics = compute_metrics(model_outs, y_true, class_names,
                                        prefix=f"{split}_",
                                        suffix=f"_{labels_set[k].lower()}")
                model_outs_all.append(model_outs)
                ys_all.append(y_true)
                if config['wandb_api_key']:
                    print('logging', metrics)
                    wandb.log(metrics)

        model_outs_all = torch.cat(model_outs_all)
        ys_all = torch.cat(ys_all)
        class_names = None if "imagenet" in dataset_name else dataset.class_names
        metrics_all = compute_metrics(model_outs_all, ys_all, class_names, prefix=f"{split}_")
        if config.get('wandb_api_key', None):
            print('logging', metrics_all)
            wandb.log(metrics_all)

if __name__ == "__main__":
    main()
