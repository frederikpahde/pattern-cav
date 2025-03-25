import copy
import gc
import logging
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from crp.attribution import CondAttribution
from torch.utils.data import DataLoader
from datasets import get_dataset, get_dataset_kwargs
from model_correction import get_correction_method
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs
from utils.distance_metrics import cosine_similarities_batch
torch.random.manual_seed(0)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/cav_analysis/isic_attacked/local/vgg16_AClarc_lamb1_signal_cavs_max_sgd_lr0.0005_features.28.yaml")
    parser.add_argument('--cav_type', type=str, default="cavs_max")
    parser.add_argument('--direction_mode', type=str, default=None)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config['config_file'] = args.config_file
    if args.direction_mode:
        config["cav_mode"] = args.cav_type
        config["direction_mode"] = args.direction_mode
        
    plot_alignment(config)
    
    

def plot_alignment(config):
    """ Computes cosine similarity between CAV and actual difference between clean and artifact sample
    Args:
        config (dict): config for model correction run
    """

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name'] + "_attacked" if "attacked" not in config['dataset_name'] else config['dataset_name']
    model_name = config['model_name']
    data_paths = config['data_paths']
    batch_size = config['batch_size']
    artifact_name = config["artifact"]
    img_size = config.get('img_size', 224)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    config["device"] = device
    layer_name = config["layer_name"]
    direction_mode = config["direction_mode"]
    artifact_type = config.get("artifact_type", "intrinsic")
    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=config.get('p_artifact', 1.0),
                                        image_size=img_size,
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        **artifact_kwargs, **dataset_specific_kwargs)
    
    dataset_clean = get_dataset(dataset_name)(data_paths=data_paths,
                                              normalize_data=True,
                                              p_artifact=0,
                                              image_size=img_size,
                                              artifact_type=config.get('artifact_type', None),
                                              attacked_classes=[],
                                              **artifact_kwargs, **dataset_specific_kwargs)

    n_classes = len(dataset.class_names)
    ckpt_path = config['ckpt_path']

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device)

    # Construct correction kwargs
    method = "Clarc"
    kwargs_correction = {}
    artifact_idxs_train = [i for i in dataset.idxs_train if i in dataset.sample_ids_by_artifact[config['artifact']]]
    kwargs_correction['class_names'] = dataset.class_names
    kwargs_correction['artifact_sample_ids'] = artifact_idxs_train
    kwargs_correction['sample_ids'] = dataset.idxs_train
    kwargs_correction['mode'] = config["cav_mode"]

    correction_method = get_correction_method(method)
    model_corrected = correction_method(copy.deepcopy(model), config, **kwargs_correction)

    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

    cav = model_corrected.cav.clone().detach().cpu().reshape(-1).numpy()
    del model_corrected
    torch.cuda.empty_cache();
    gc.collect()
    model.eval()
    attribution = CondAttribution(model)

    for split in [
        # 'train',
        'test',
        # 'val'
    ]:
        split_set = sets[split]

        dataset_split = dataset.get_subset_by_idxs(split_set)
        dataset_clean_split = dataset_clean.get_subset_by_idxs(split_set)

        dataset_artifact_only = dataset_split.get_subset_by_idxs(dataset_split.artifact_ids)
        dataset_artifact_only_clean = dataset_clean_split.get_subset_by_idxs(dataset_split.artifact_ids)

        dl_art = DataLoader(dataset_artifact_only, batch_size=batch_size, shuffle=False)
        dl_clean = DataLoader(dataset_artifact_only_clean, batch_size=batch_size, shuffle=False)

        similarities_all = None

        diffs = []

        num_samples = 10
        high_alignment = []
        high_alignment_imgs = []
        low_alignment = []
        low_alignment_imgs = []

        for (x_att, _), (x_clean, _) in zip(tqdm.tqdm(dl_art), dl_clean):

            # Compute latent representation
            x_latent_att = get_features(x_att.to(device), config, attribution).detach().cpu()
            x_latent_clean = get_features(x_clean.to(device), config, attribution).detach().cpu()

            # Compute similarities between representation difference (attacked-clean) and CAV
            diff_latent = (x_latent_att - x_latent_clean)

            # mean_cav += diff_latent.sum(0).reshape(-1) / len(dataset_artifact_only)
            diff_flat = diff_latent.numpy().reshape(diff_latent.shape[0], -1)
            diffs.append(diff_flat)

            similarities = cosine_similarities_batch(diff_flat, cav)
            similarities_all = similarities if similarities_all is None else np.concatenate(
                [similarities_all, similarities])

            def torch2numpy(x):
                std = np.array(dataset.normalize_fn.std) if dataset.normalize_fn else torch.ones(3)
                mean = np.array(dataset.normalize_fn.mean) if dataset.normalize_fn else torch.zeros(3)
                x_new = x.detach().cpu().permute(0, 2, 3, 1).numpy() * std[None] + mean[None]
                return x_new.clip(min=0.0, max=1.0)

            similarities_sorted = torch.sort(torch.tensor(similarities), descending=True)
            high_alignment.append(similarities_sorted.values[:num_samples])
            high_alignment_imgs.append(torch2numpy(x_att[similarities_sorted.indices[:num_samples]]))
            similarities_sorted = torch.sort(torch.tensor(similarities).abs(), descending=False)
            low_alignment.append(similarities_sorted.values[:num_samples])
            low_alignment_imgs.append(torch2numpy(x_att[similarities_sorted.indices[:num_samples]]))

            similarities_all = similarities if similarities_all is None else np.concatenate(
                [similarities_all, similarities])
        
        ### PLOT high_alignment_imgs and low_alignment_imgs
        path_img_alignment = f"results/cav_alignment/{dataset_name}_{artifact_name}_{artifact_type}/{model_name}/{split}_{layer_name}_{direction_mode}_alignment"
        _ = [plot_aligned_samples(high_alignment_imgs, low_alignment_imgs, high_alignment, low_alignment,
                                num_samples, f"{path_img_alignment}.{ending}") for ending in ["png"]]


def visualize_dataset(ds, path, start_idx):
    nrows = 4
    ncols = 5
    size = 3

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for i in range(nrows * ncols):
        ax = axs[i // ncols][i % ncols]
        idx = start_idx + i
        img, y = ds[idx]
        img = np.moveaxis(ds.reverse_normalization(img).numpy(), 0, 2)
        ax.imshow(img)
        ax.set_title(y)
        # ax.axis('off')
    fig.savefig(path)

def plot_aligned_samples(high_alignment_imgs, low_alignment_imgs, high_alignment, low_alignment,
                         num_samples, path_img_alignment):
    # concat all images
    high_alignment_imgs = np.concatenate(high_alignment_imgs, axis=0)
    low_alignment_imgs = np.concatenate(low_alignment_imgs, axis=0)

    # concat alignment scores
    high_alignment = np.concatenate(high_alignment, axis=0)
    low_alignment = np.concatenate(low_alignment, axis=0)

    # sort images by alignment
    high_alignment_imgs = high_alignment_imgs[high_alignment.argsort()][::-1][:num_samples]
    low_alignment_imgs = low_alignment_imgs[low_alignment.argsort()][:num_samples]
    high_alignment = high_alignment[high_alignment.argsort()][::-1][:num_samples]
    low_alignment = low_alignment[low_alignment.argsort()][:num_samples]

    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        axs[0, i].set_title(f"{high_alignment[i]:.2f}")
        axs[1, i].set_title(f"{low_alignment[i]:.2f}")
        axs[0, i].imshow(high_alignment_imgs[i])
        axs[1, i].imshow(low_alignment_imgs[i])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

    axs[0, 0].set_ylabel("high alignment")
    axs[1, 0].set_ylabel("low alignment")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path_img_alignment), exist_ok=True)
    plt.savefig(path_img_alignment, dpi=300)
    plt.show()
    plt.close()


def get_features(batch, config, attribution):

    batch.requires_grad = True
    dummy_cond = [{"y": 0} for _ in range(len(batch))]
    record_layer=[config["layer_name"]]
    attr = attribution(batch.to(config["device"]), dummy_cond, record_layer=record_layer)
    if config["cav_mode"] == "cavs_full":
        features = attr.activations[config["layer_name"]]
    else:
        features = attr.activations[config["layer_name"]].flatten(start_dim=2).max(2)[0]
    return features


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
