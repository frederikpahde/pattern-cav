import copy
import gc
import logging
import os
from argparse import ArgumentParser

import numpy as np
import scipy
import torch
import torchvision.transforms as T
import tqdm
import wandb
import yaml
from crp.attribution import CondAttribution
from torch.utils.data import DataLoader

from datasets import get_dataset
from model_correction import get_correction_method
from models import get_fn_model_loader
from utils.distance_metrics import cosine_similarities_batch

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', default="config_files/real_artifacts_clarc/isic/local/vgg16_band_aid_RRClarc_lamb100000_signal_cavs_max_sgd_lr0.0005_features.28.yaml")
    parser.add_argument("--artifact", default="band_aid")

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

    config['attack_artifact'] = args.artifact
    config['config_file'] = args.config_file

    method = config.get("method", "")
    if "aclarc" in method.lower():
        if args.artifact in args.config_file:
            measure_quality_cav_auto(config)
        else:
            logger.info(f"Skipping eval wrt {args.artifact} for {args.config_file}")
        
    else:
        logger.info(f"Skipping quality-of-CAV metric for method {method}")


def measure_quality_cav_auto(config):
    """ Computes cosine similarity between CAV and actual difference between clean and artifact sample
    Args:
        config (dict): config for model correction run
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = config['dataset_name']
    model_name = config['model_name']
    batch_size = config['batch_size']
    artifact_name = config["attack_artifact"]
    config["device"] = device

    dataset = get_dataset(f"{dataset_name}")(data_paths=config['data_paths'],
                                             normalize_data=True,
                                             artifact_ids_file=config['artifacts_file'],
                                             artifact=config['attack_artifact'])

    dataset_hm = get_dataset(f"{dataset_name}_hm")(data_paths=config['data_paths'],
                                                   normalize_data=True,
                                                   artifact_ids_file=config['artifacts_file'],
                                                   artifact=config['attack_artifact'])

    n_classes = len(dataset.class_names)
    ckpt_path = config['ckpt_path']

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device)

    # Construct correction kwargs
    method = config["method"]
    kwargs_correction = {}
    if "clarc" in method.lower():
        kwargs_correction['class_names'] = dataset.class_names
        kwargs_correction['artifact_sample_ids'] = dataset.sample_ids_by_artifact[config['artifact']]
        kwargs_correction['sample_ids'] = np.array([i for i in dataset.idxs_train])  # [i for i in dataset.idxs_val]
        kwargs_correction['mode'] = config["cav_mode"]

    correction_method = get_correction_method(method)
    model_corrected = correction_method(copy.deepcopy(model), config, **kwargs_correction)

    ### COLLECT ARTIFACTS
    gaussian = T.GaussianBlur(kernel_size=41, sigma=5.0)
    artifact_samples = dataset_hm.sample_ids_by_artifact[config['attack_artifact']]

    masks = []
    artifacts = []
    batch_size = config['batch_size']
    print(f"There are {len(artifact_samples)} artifact samples")
    for k, samples in enumerate([artifact_samples]):

        n_samples = len(samples)
        n_batches = int(np.ceil(n_samples / batch_size))

        for i in tqdm.tqdm(range(n_batches)):
            samples_batch = samples[i * batch_size:(i + 1) * batch_size]
            data = torch.stack([dataset_hm[j][0] for j in samples_batch], dim=0)
            mask = torch.stack([dataset_hm[j][2] for j in samples_batch])
            mask = gaussian(mask.clamp(min=0)) ** 1.0
            mask = mask / mask.abs().flatten(start_dim=1).max(1)[0][:, None, None]
            artifacts.append(data)
            masks.append(mask)

    masks = torch.cat(masks, 0)
    artifacts = torch.cat(artifacts, 0)

    ### Iterate over clean samples
    clean_sample_ids = dataset.clean_sample_ids

    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

    cav = model_corrected.cav.clone().detach().cpu().reshape(-1).numpy()
    del model_corrected
    torch.cuda.empty_cache();
    gc.collect()

    attribution = CondAttribution(model)
    rng = np.random.default_rng(0)

    results = {}
    for split in [
        # 'train', 
        'test', 
        'val'
        ]:
        split_set = sets[split]
        split_set_clean = [i for i in split_set if i in clean_sample_ids]

        dataset_clean_split = dataset.get_subset_by_idxs(split_set_clean)

        dl = DataLoader(dataset_clean_split, batch_size=batch_size, shuffle=False)

        similarities_all = None
        scores_clean = []
        scores_attacked = []
        mean_cav = torch.zeros_like(torch.tensor(cav))
        for x_batch, _ in tqdm.tqdm(dl):
            # Compute latent representation (clean)
            x_latent = get_features(x_batch.to(device), config, attribution).detach().cpu()

            # Insert random artifact
            pick = rng.choice(range(len(masks)), len(x_batch))
            m = masks[pick][:, None, :, :]
            artifact = artifacts[pick]
            x_batch_attacked = x_batch * (1 - m) + artifact * m

            # Compute latent representation (attacked)
            x_latent_attacked = get_features(x_batch_attacked.to(device), config,
                                             attribution).detach().cpu()

            # Compute similarities between representation difference (attacked-clean) and CAV
            diff_latent = (x_latent_attacked - x_latent)
            diff_flat = diff_latent.numpy().reshape(diff_latent.shape[0], -1)
            mean_cav += diff_latent.sum(0).reshape(-1) / len(dataset_clean_split)
            similarities = cosine_similarities_batch(diff_flat, cav)

            score_attacked = (x_latent_attacked.flatten(start_dim=1).numpy() * cav[None]).sum(1)
            score_clean = (x_latent.flatten(start_dim=1).numpy() * cav[None]).sum(1)
            scores_clean.append(score_clean)
            scores_attacked.append(score_attacked)

            similarities_all = similarities if similarities_all is None else np.concatenate(
                [similarities_all, similarities])

        scores_clean = np.concatenate(scores_clean)
        scores_attacked = np.concatenate(scores_attacked)

        thresh = np.linspace(scores_clean.min(), scores_attacked.max(), 1000)
        tpr = []
        fpr = []
        for t in thresh:
            tpr.append((scores_attacked > t).mean())
            fpr.append((scores_clean > t).mean())

        auc = - np.trapz(tpr, fpr)
        print(f"AUC: {auc}")
        results[f"cav_auc_{artifact_name}_{split}"] = auc


        results[f"cav_similarity_{artifact_name}_{split}_mean_cav"] = cosine_similarities_batch(mean_cav[None], cav).flatten()
        results[f"cav_similarity_{artifact_name}_{split}"] = similarities_all.mean()
        results[f"cav_similarity_abs_{artifact_name}_{split}"] = np.abs(similarities_all).mean()
        results[f"cav_similarity_{artifact_name}_{split}_stderr"] = similarities_all.std() / similarities_all.shape[
            0] ** 0.5

    if config.get('wandb_api_key', None):
        wandb.log(results)


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
