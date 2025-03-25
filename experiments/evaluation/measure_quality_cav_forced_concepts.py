import copy
import gc
import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
import wandb
import yaml
from crp.attribution import CondAttribution
from torch.utils.data import DataLoader
from sklearn import metrics
from datasets import get_dataset
from models import get_fn_model_loader
from utils.distance_metrics import cosine_similarities_batch
from utils.se import get_se_auc
torch.random.manual_seed(0)
from model_training.train_concept_bank import FORCED_CONCEPTS

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/tcav_experiments/funnybirds_forced_concept/local/vgg16_funnybirds_attribute_phcb_model_signal-1d-features.28.yaml")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config_name = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config['config_file'] = args.config_file
    config["config_name"] = config_name
    config["wandb_id"] = config_name

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    
    measure_quality_cav_forced_concept(config)


def measure_quality_cav_forced_concept(config):
    """ Computes cosine similarity between CAV and actual difference between original and manipulated FunnyBirds samples
    Args:
        config (dict): config for model correction run
    """

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    
    dataset_name = config['dataset_name']
    dataset_name_attribute = config['dataset_name_attribute']

    model_name = config['model_name']

    data_paths = config['data_paths']
    batch_size = config['batch_size']
    img_size = config.get('img_size', 224)

    model_savedir = config['model_savedir']

    config["device"] = device


    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        image_size=img_size)

    n_classes = len(dataset.class_names)
    ckpt_path = config["ckpt_path"]

    concept_bank = torch.load(f"{model_savedir}/{dataset_name_attribute}/{config['config_name']}/concept_bank.pt")
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device)

    sets = {
        'train': dataset.idxs_train,
        'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }

    attribution = CondAttribution(model)

    results = {}
    for split in [
        'test',
    ]:
        print("Starting with split", split)
        split_set = sets[split]

        all_aucs = []
        similarities_all_concepts = None

        auc_y_all = None
        for cname, cav in concept_bank.items():
            if not cname in FORCED_CONCEPTS.keys():
                logger.info(f"Skipping evaluation of concept {cname}")
                continue
            print(cname)
            cav = cav.numpy()
            affected_part = cname.split("::")[0]
            dataset_wo_concept = get_dataset(dataset_name)(data_paths=data_paths,
                                                                    normalize_data=True,
                                                                    image_size=img_size,
                                                                    subfolder_name_extension=f"_random_{affected_part}")
        

            dataset_split = dataset.get_subset_by_idxs(split_set)
            dataset_wo_concept_split = dataset_wo_concept.get_subset_by_idxs(split_set)

            pos_sample_ids = np.where(dataset_split.metadata.label.values == FORCED_CONCEPTS[cname])[0]

            dataset_pos_samples = dataset_split.get_subset_by_idxs(pos_sample_ids)
            dataset_pos_samples_wo_concept = dataset_wo_concept_split.get_subset_by_idxs(pos_sample_ids)

            dl_clean = DataLoader(dataset_pos_samples, batch_size=batch_size, shuffle=False)
            dl_random = DataLoader(dataset_pos_samples_wo_concept, batch_size=batch_size, shuffle=False)

            similarities_all = None

            diffs = []

            scores_clean = []
            scores_random = []

            for (x_clean, _), (x_random, _) in zip(tqdm.tqdm(dl_clean), dl_random):

                # Compute latent representation
                x_latent_random = get_features(x_random.to(device), config, attribution).detach().cpu()
                x_latent_clean = get_features(x_clean.to(device), config, attribution).detach().cpu()

                # Compute similarities between representation difference (attacked-clean) and CAV
                diff_latent = (x_latent_clean - x_latent_random)
                diff_flat = diff_latent.numpy().reshape(diff_latent.shape[0], -1)
                diffs.append(diff_flat)
                similarities = cosine_similarities_batch(diff_flat, cav)
                similarities_all = similarities if similarities_all is None else np.concatenate(
                    [similarities_all, similarities])

                # For AUC computation
                score_random = (x_latent_random.flatten(start_dim=1).numpy() * cav[None]).sum(1)
                score_clean = (x_latent_clean.flatten(start_dim=1).numpy() * cav[None]).sum(1)
                scores_clean.append(score_clean)
                scores_random.append(score_random)

                similarities_all = similarities if similarities_all is None else np.concatenate(
                    [similarities_all, similarities])
                
            scores_clean = np.concatenate(scores_clean)
            scores_random = np.concatenate(scores_random)

            pred = np.concatenate([scores_random, scores_clean])
            y = np.concatenate([np.zeros_like(scores_random), np.ones_like(scores_clean)])
            auc_y_all = y if auc_y_all is None else np.concatenate([auc_y_all, y])
            print(f"Pos: {len(scores_random)}, Neg: {len(scores_clean)}")
            fpr, tpr, _ = metrics.roc_curve(y, pred)
            auc = metrics.auc(fpr, tpr)
            auc_se = get_se_auc(auc, (y==1).sum(), (y==0).sum())

            all_aucs.append(auc)
            similarities_all_concepts = similarities_all if similarities_all_concepts is None else np.concatenate(
                    [similarities_all_concepts, similarities_all])

            results[f"cav_auc_{cname}_{split}"] = auc
            results[f"cav_auc_{cname}_{split}_se"] = auc_se
            results[f"cav_similarity_{cname}_{split}"] = similarities_all.mean()
            results[f"cav_similarity_{cname}_{split}_stderr"] = similarities_all.std() / similarities_all.shape[
                0] ** 0.5
            
        auc_mean = np.array(all_aucs).mean()
        results[f"cav_auc_mean_{split}"] = auc_mean
        results[f"cav_auc_mean_{split}_se"] = get_se_auc(auc_mean, (auc_y_all == 1).sum(), (auc_y_all == 0).sum())
        results[f"cav_similarity_mean_{split}"] =similarities_all_concepts.mean()
        results[f"cav_similarity_mean_{split}_se"] = similarities_all_concepts.std() / similarities_all_concepts.shape[
                0] ** 0.5
        
    wandb.log({**results, **config})


def get_features(batch, config, attribution):

    batch.requires_grad = True
    dummy_cond = [{"y": 0} for _ in range(len(batch))]
    record_layer=[config["layer_name"]]
    attr = attribution(batch.to(config["device"]), dummy_cond, record_layer=record_layer)
    features = attr.activations[config["layer_name"]]
    if config["cav_dim"] == 1:
        features = features if features.dim() == 2 else features.flatten(start_dim=2).max(2)[0]
    return features


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
