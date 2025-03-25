import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
import yaml
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from zennit.composites import EpsilonPlusFlat
from zennit import image as zimage
from datasets import get_dataset, get_dataset_kwargs
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader, get_canonizer
from utils.artificial_artifact import get_artifact_kwargs

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    parser = ArgumentParser()
    # parser.add_argument("--sample_ids", default="18561,18925,20119,2671", type=str)
    parser.add_argument("--sample_ids", default=None, type=str)
    parser.add_argument("--normalized", default="max", type=str)
    parser.add_argument('--config_file',
                        default="config_files/real_artifacts_clarc/isic/local/vgg16_band_aid_RRClarc_lamb1000000_svm_cavs_max_sgd_lr0.0005_features.28.yaml")
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

    config['sample_ids'] = [int(i) for i in args.sample_ids.split(",")] if args.sample_ids else None
    config['config_file'] = args.config_file
    config['normalized'] = args.normalized

    plot_corrected_model(config)


def plot_corrected_model(config):
    dataset_name = config['dataset_name']
    config_name = os.path.basename(config["config_file"])[:-5]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs_data = {
        "p_artifact": 1.0,
        "artifact_type": config['artifact_type'],
        "attacked_classes": config['attacked_classes']
    } if config['artifact'] == 'artificial' else {
        "artifact_ids_file": config.get('artifacts_file', None)
    }

    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)

    data_paths=config['data_paths']

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        **kwargs_data, **artifact_kwargs, **dataset_specific_kwargs)

    sample_ids = config['sample_ids']
    if sample_ids is None:
        # only "corrected" artifact
        sample_ids = dataset.sample_ids_by_artifact[config['artifact']][:8]

        # all artifacts
        # sample_ids = [sid for _, sids in dataset.sample_ids_by_artifact.items() for sid in sids[:3]]

    data = torch.stack([dataset[j][0] for j in sample_ids], dim=0).to(device)

    target = torch.stack([dataset[j][1] for j in sample_ids], dim=0)
    ckpt_path_corrected = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    if config["num_epochs"] == 0 and dataset_name == "imagenet":
        ckpt_path_corrected = None
    ckpt_path_original = config['ckpt_path']
    model_corrected = get_fn_model_loader(model_name=config['model_name'])(n_class=len(dataset.class_names),
                                                                           ckpt_path=ckpt_path_corrected)
    model_corrected = prepare_model_for_evaluation(model_corrected, dataset, ckpt_path_corrected, device, config)

    model_original = get_fn_model_loader(model_name=config['model_name'])(n_class=len(dataset.class_names),
                                                                           ckpt_path=ckpt_path_original)

    model_original.eval()
    model_original = model_original.to(device)

    attribution_corrected = CondAttribution(model_corrected)
    attribution_original = CondAttribution(model_original)
    canonizers = get_canonizer(config['model_name'])
    composite = EpsilonPlusFlat(canonizers)

    condition = [{"y": c_id.item()} for c_id in target]
    attr_corrected = attribution_corrected(data.requires_grad_(), condition, composite)

    max = get_normalization_constant(attr_corrected, config['normalized'])

    heatmaps_corrected = attr_corrected.heatmap / max
    heatmaps_corrected = heatmaps_corrected.detach().cpu().numpy()

    # computed corrupted heatmaps
    condition_original = [{"y": c_id.item()} for c_id in target]
    attr_original = attribution_original(data.requires_grad_(), condition_original, composite)

    max = get_normalization_constant(attr_original, config['normalized'])

    heatmaps_original = attr_original.heatmap / max
    heatmaps_original = heatmaps_original.detach().cpu().numpy()

    heatmaps_diff = heatmaps_corrected - heatmaps_original
    if config['normalized'] == "max":
        heatmaps_diff /= heatmaps_diff.reshape(heatmaps_diff.shape[0], -1).max(1)[:, None, None]
    elif config['normalized'] == "abs_max":
        heatmaps_diff /= np.abs(heatmaps_diff).reshape(heatmaps_diff.shape[0], -1).max(1)[:, None, None]
    # plot input images and heatmaps in grid
    size = 2
    fig, axs = plt.subplots(4, len(sample_ids), figsize=(len(sample_ids) * size, 3 * size), dpi=300)

    for i, sample_id in enumerate(sample_ids):
        axs[0, i].imshow(dataset.reverse_normalization(dataset[sample_id][0]).permute(1, 2, 0) / 255)

        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[0, i].set_title(f"Sample {sample_id}")
        # axs[0, i].axis("off")

        axs[1, i].imshow(heatmaps_original[i], vmin=-1, vmax=1, cmap="bwr")
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        # axs[1, i].axis("off")

        axs[2, i].imshow(heatmaps_corrected[i], vmin=-1, vmax=1, cmap="bwr")
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        # axs[2, i].axis("off")

        axs[3, i].imshow(zimage.imgify(heatmaps_diff[i], vmin=-1., vmax=1., level=1.0, cmap='coldnhot'))
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])
        # axs[3, i].axis("off")

        # make border thicker
        for ax in axs[:, i]:
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        # set label for the first column
        if i == 0:
            axs[0, i].set_ylabel("Input")
            axs[1, i].set_ylabel("Vanilla")
            axs[2, i].set_ylabel(str(config['method']))
            axs[3, i].set_ylabel("Difference")

    plt.tight_layout()

    # save figure with and without labels as pdf
    path = f"results/plot_corrected_model"
    if not os.path.exists(path):
        os.makedirs(path)

    # plt.savefig(f"{path}/{config['wandb_id']}.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{path}/{config['wandb_id']}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{path}/{config['wandb_id']}.jpeg", bbox_inches="tight", dpi=300)

    # disable labels
    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        # ax.axis("off")
    # plt.savefig(f"{path}/{config['wandb_id']}_no_labels.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.savefig(f"{path}/{config['wandb_id']}_no_labels.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{path}/{config['wandb_id']}_no_labels.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # log png to wandb
    wandb.log({"corrected_model": wandb.Image(f"{path}/{config['wandb_id']}.jpeg")})

    print("Done.")


def get_normalization_constant(attr, normalization_mode):
    if normalization_mode == 'max':
        return attr.heatmap.flatten(start_dim=1).max(1, keepdim=True).values[:, None]
    elif normalization_mode == 'abs_max':
        return attr.heatmap.flatten(start_dim=1).abs().max(1, keepdim=True).values[:, None]
    else:
        raise ValueError("Unknown normalization")

if __name__ == "__main__":
    main()
