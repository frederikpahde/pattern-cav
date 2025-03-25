import copy
import gc
import logging
import os
from argparse import ArgumentParser

import torch
import tqdm
import wandb
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs
from experiments.evaluation.compute_metrics import aggregate_tcav_metrics, compute_tcav_metrics_batch
from model_correction.clarc import Clarc
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/clarc/isic_attacked/local/vgg16_RRClarc_lamb100000_signal_cavs_max_sgd_lr0.0005_features.28.yaml")
    parser.add_argument('--cav_type', type=str, default=None)
    parser.add_argument('--direction_type', type=str, default=None)
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--before_correction', action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args.config_file)
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
    if args.cav_type:
        cav_mode = args.cav_type
        direction_mode = args.direction_type
        method = args.method
        config['method'] = method
        config["lamb"] = 1
        config["cav_mode"] = cav_mode
        config["direction_mode"] = direction_mode


    method = config.get("method", "")
    if not "vanilla" in method.lower():
        measure_quality_cav(config, args.before_correction)
    else:
        logger.info(f"Skipping TCAV metric for method {method}")

def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

def measure_quality_cav(config, before_correction):
    """ Computes TCAV scores
    Args:
        config (dict): config for model correction run
    """

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name']
    model_name = config['model_name']

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
                                        # p_artifact=1.0,
                                        artifact_ids_file=artifacts_file,
                                        image_size=img_size,
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        **artifact_kwargs, **dataset_specific_kwargs)
    
    if "funnybirds_ch" in dataset_name:
        base_path ="/".join(data_paths[-1].split("/")[:-1])
        data_paths[-1] = f"{base_path}/test_clean"

    n_classes = len(dataset.class_names)
    # ckpt_path = config['ckpt_path']
    config_name = os.path.basename(config['config_file'])[:-5]
    
    if before_correction:
        ckpt_path = config["ckpt_path"]
        print(f"Using uncorrected model ckpt: {ckpt_path}")
    else:
        ckpt_path = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"

    if config["num_epochs"] == 0 and dataset_name == "imagenet":
        ckpt_path = None

    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device)

    # Construct correction kwargs
    kwargs_correction = {}
    artifact_idxs_train = [i for i in dataset.idxs_train if i in dataset.sample_ids_by_artifact[config['artifact']]]
    kwargs_correction['class_names'] = dataset.class_names
    kwargs_correction['artifact_sample_ids'] = artifact_idxs_train
    kwargs_correction['sample_ids'] = dataset.idxs_train
    kwargs_correction['mode'] = config.get("cav_mode", "cavs_max")

    model_corrected = Clarc(copy.deepcopy(model), config, **kwargs_correction)

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

    results = {}
    for split in [
        'train',
        'test',
        'val'
    ]:
        split_set = sets[split]

        artifact_ids_split = [i for i in dataset.sample_ids_by_artifact[artifact_name] if i in split_set]
        dataset_artifact_only = dataset.get_subset_by_idxs(artifact_ids_split)

        dl_art = DataLoader(dataset_artifact_only, batch_size=1, shuffle=False)

        
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
        for sample in tqdm.tqdm(dl_art):
            x_att, y = sample
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

                tcav_metrics_batch = compute_tcav_metrics_batch(grad, cav)
                        
                TCAV_pos += tcav_metrics_batch["TCAV_pos"]
                TCAV_neg += tcav_metrics_batch["TCAV_neg"]
                TCAV_pos_mean += tcav_metrics_batch["TCAV_pos_mean"]
                TCAV_neg_mean += tcav_metrics_batch["TCAV_neg_mean"]

                TCAV_sens_list.append(tcav_metrics_batch["TCAV_sensitivity"])

        

        tcav_metrics = aggregate_tcav_metrics(TCAV_pos, TCAV_neg, TCAV_pos_mean, TCAV_neg_mean, TCAV_sens_list)

        metric_extension = "_before_correction" if before_correction else ""
        results[f"{split}_mean_tcav_quotient{metric_extension}"] = tcav_metrics['mean_tcav_quotient']
        results[f"{split}_mean_quotient_sderr{metric_extension}"] = tcav_metrics['mean_quotient_sderr']

        results[f"{split}_tcav_quotient{metric_extension}"] = tcav_metrics['tcav_quotient']
        results[f"{split}_quotient_sderr{metric_extension}"] = tcav_metrics['quotient_sderr']

        results[f"{split}_mean_tcav_sensitivity{metric_extension}"] = tcav_metrics['mean_tcav_sensitivity']
        results[f"{split}_mean_tcav_sensitivity_sem{metric_extension}"] = tcav_metrics['mean_tcav_sensitivity_sem']

        if config.get('wandb_api_key', None):
            wandb.log({**results, **config})


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
