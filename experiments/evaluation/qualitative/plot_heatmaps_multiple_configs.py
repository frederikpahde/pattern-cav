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
    parser.add_argument('--config_files',
                        default=[
                            ## Band-aid
                            "config_files/real_artifacts_clarc/isic/local/vgg16_band_aid_RRClarc_lamb1000000_svm_cavs_max_sgd_lr0.0005_features.28.yaml",
                            "config_files/real_artifacts_clarc/isic/local/vgg16_band_aid_RRClarc_lamb1000000_signal_cavs_max_sgd_lr0.0005_features.28.yaml"
                            ]
                            )
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    configs = []
    for config_file in args.config_files:
        with open(config_file, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                config["wandb_id"] = os.path.basename(config_file)[:-5]
            except yaml.YAMLError as exc:
                print(exc)
                config = {}
        


        config['sample_ids'] = [int(i) for i in args.sample_ids.split(",")] if args.sample_ids else None
        config['config_file'] = config_file
        config['normalized'] = args.normalized
        configs.append(config)

    plot_corrected_model(configs)


def plot_corrected_model(configs):
    """Plot attribution heatmaps for multiple CAV variants, as well as the difference heatmap in comparison with Vanilla

    Args:
        configs (array[dict]): list of experiment configs
    """
    dataset_name = configs[0]['dataset_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs_data = {
        "p_artifact": 1.0,
        "artifact_type": configs[0]['artifact_type'],
        "attacked_classes": configs[0]['attacked_classes']
    } if configs[0]['artifact'] == 'artificial' else {
        "artifact_ids_file": configs[0].get('artifacts_file', None)
    }

    artifact_kwargs = get_artifact_kwargs(configs[0])
    dataset_specific_kwargs = get_dataset_kwargs(configs[0])

    data_paths=configs[0]['data_paths']

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        **kwargs_data, **artifact_kwargs, **dataset_specific_kwargs)

    sample_ids = configs[0]['sample_ids']
    if sample_ids is None:
        # only "corrected" artifact
        sample_ids = dataset.sample_ids_by_artifact[configs[0]['artifact']][:8]

    data = torch.stack([dataset[j][0] for j in sample_ids], dim=0).to(device)

    target = torch.stack([dataset[j][1] for j in sample_ids], dim=0)

    config_names = [os.path.basename(config["config_file"])[:-5] for config in configs]
    ckpt_paths_corrected = [f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt" for config, config_name in zip(configs, config_names)]
    if configs[0]["num_epochs"] == 0 and dataset_name == "imagenet":
        ckpt_paths_corrected = []
    ckpt_path_original = configs[0]['ckpt_path']

    models_corrected = []
    for ckpt_path_corrected, config in zip(ckpt_paths_corrected, configs):
        model_corrected = get_fn_model_loader(model_name=configs[0]['model_name'])(n_class=len(dataset.class_names),
                                                                            ckpt_path=ckpt_path_corrected)
        model_corrected = prepare_model_for_evaluation(model_corrected, dataset, ckpt_path_corrected, device, config)
        models_corrected.append(model_corrected)

    model_original = get_fn_model_loader(model_name=configs[0]['model_name'])(n_class=len(dataset.class_names),
                                                                           ckpt_path=ckpt_path_original)

    model_original.eval()
    model_original = model_original.to(device)

    attributions_corrected = [CondAttribution(model_corrected) for model_corrected in models_corrected]
    attribution_original = CondAttribution(model_original)
    canonizers = get_canonizer(config['model_name'])
    composite = EpsilonPlusFlat(canonizers)

    condition = [{"y": c_id.item()} for c_id in target]
    attrs_corrected = [attribution_corrected(data.requires_grad_(), condition, composite) for attribution_corrected in attributions_corrected]

    maxs = [get_normalization_constant(attr_corrected, config['normalized']) for attr_corrected in attrs_corrected]

    heatmaps_corrected = [(attr_corrected.heatmap / m).detach().cpu().numpy() for attr_corrected, m in zip (attrs_corrected, maxs)]

    condition_original = [{"y": c_id.item()} for c_id in target]
    attr_original = attribution_original(data.requires_grad_(), condition_original, composite)

    max = get_normalization_constant(attr_original, config['normalized'])
    heatmap_original = attr_original.heatmap / max
    heatmap_original = heatmap_original.detach().cpu().numpy()

    heatmaps_diffs = [heatmap_corrected - heatmap_original for heatmap_corrected in heatmaps_corrected]
    if config['normalized'] == "max":
        m = np.array([heatmaps_diff.reshape(heatmaps_diff.shape[0], -1).max(1) for heatmaps_diff in heatmaps_diffs]).max(0)[:, None, None]
        
    elif config['normalized'] == "abs_max":
        m = np.array([np.abs(heatmaps_diff).reshape(heatmaps_diff.shape[0], -1).max(1) for heatmaps_diff in heatmaps_diffs]).max(0)[:, None, None]
    
    heatmaps_diffs = [heatmaps_diff / m for heatmaps_diff in heatmaps_diffs]
    # plot input images and heatmaps in grid
    size = 2
    fig, axs = plt.subplots(6, len(sample_ids), figsize=(len(sample_ids) * size, 6 * size), dpi=300)

    for i, sample_id in enumerate(sample_ids):
        axs[0, i].imshow(dataset.reverse_normalization(dataset[sample_id][0]).permute(1, 2, 0) / 255)

        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[0, i].set_title(f"Sample {sample_id}")
        # axs[0, i].axis("off")

        axs[1, i].imshow(heatmap_original[i], vmin=-1, vmax=1, cmap="bwr")
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        # axs[1, i].axis("off")

        axs[2, i].imshow(heatmaps_corrected[0][i], vmin=-1, vmax=1, cmap="bwr")
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])

        
        # axs[2, i].axis("off")

        axs[3, i].imshow(zimage.imgify(heatmaps_diffs[0][i], vmin=-1., vmax=1., level=1.0, cmap='coldnhot'))
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])

        axs[4, i].imshow(heatmaps_corrected[1][i], vmin=-1, vmax=1, cmap="bwr")
        axs[4, i].set_xticks([])
        axs[4, i].set_yticks([])

        axs[5, i].imshow(zimage.imgify(heatmaps_diffs[1][i], vmin=-1., vmax=1., level=1.0, cmap='coldnhot'))
        axs[5, i].set_xticks([])
        axs[5, i].set_yticks([])
        # axs[3, i].axis("off")

        # make border thicker
        for ax in axs[:, i]:
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        # set label for the first column
        if i == 0:
            axs[0, i].set_ylabel("Input")
            axs[1, i].set_ylabel("Vanilla")
            axs[2, i].set_ylabel(f"{configs[0]['method']} - {configs[0]['direction_mode']}")
            axs[3, i].set_ylabel("Difference")
            axs[4, i].set_ylabel(f"{configs[1]['method']} - {configs[1]['direction_mode']}")
            axs[5, i].set_ylabel("Difference")

    plt.tight_layout()

    # save figure with and without labels as pdf
    path = f"results/plot_corrected_model"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/{configs[0]['wandb_id']}.png", bbox_inches="tight", dpi=300)

    # disable labels
    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        # ax.axis("off")

    plt.savefig(f"{path}/{configs[0]['wandb_id']}_no_labels.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{path}/{configs[0]['wandb_id']}_no_labels.pdf", bbox_inches="tight", dpi=300)
    plt.show()

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
