import logging
import os
from argparse import ArgumentParser
from experiments.evaluation.compute_metrics import aggregate_tcav_metrics, compute_tcav_metrics_batch

from model_training.train_concept_bank import FORCED_CONCEPTS
import numpy as np
import torch
import tqdm
import wandb
import yaml
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from zennit.image import imgify
from datasets import get_dataset
from models import get_fn_model_loader

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/tcav_experiments/funnybirds_forced_concept/local/vgg16_funnybirds_attribute_phcb_model_signal-1d-features.28.yaml")

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args.config_file)
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config["wandb_id"] = os.path.basename(args.config_file)[:-5]
            config['config_name'] = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    config['config_file'] = args.config_file

    # Dont run for each config
    if config["cav_type"] == "signal":
        measure_quality_cav_gt(config)
    else:
        logger.info(f"Skipping GT TCAV metric for method {config['cav_type']}")

def get_activation(module, input_, output_):
            global activations
            activations = output_
            return output_.clone()

def measure_quality_cav_gt(config):
    """ Computes TCAV scores
    Args:
        config (dict): config for model correction run
    """

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name']
    dataset_name_attribute = config['dataset_name_attribute']
    model_name = config['model_name']
    cav_mode = config.get("cav_mode", "cavs_max")
    data_paths = config['data_paths']
    img_size = config.get('img_size', 224)
    config["device"] = device
    model_savedir = config['model_savedir']

    dataset = get_dataset(dataset_name_attribute)(data_paths=data_paths,
                                        normalize_data=True,
                                        image_size=img_size)

    n_classes = len(dataset.class_names)
    ckpt_path = config["ckpt_path"]

    concept_bank = torch.load(f"{model_savedir}/{dataset_name_attribute}/{config['config_name']}/concept_bank.pt")
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device).eval()


    concept_index_pos, concept_index_neg = dataset.build_pos_neg_concept_indexes()



    results = {}
    for  split, idxs_split in {
        # 'train': dataset.idxs_train,
        # 'val': dataset.idxs_val,
        'test': dataset.idxs_test,
    }.items():
        print("Starting with split", split)

        preds = {}
        preds_no_concept = {}
        ys_all = {}

        for cname, cav in concept_bank.items():
            print(f"Eval {cname}")
            print(".")
            if not cname in FORCED_CONCEPTS.keys():
                logger.info(f"Skipping evaluation of concept {cname}")
                continue
            
            preds[cname] = []
            preds_no_concept[cname] = []
            ys_all[cname] = []

            affected_part = cname.split("::")[0]

            dataset_wo_concept = get_dataset(dataset_name_attribute)(data_paths=data_paths,
                                                                    normalize_data=True,
                                                                    image_size=img_size,
                                                                    subfolder_name_extension=f"_random_{affected_part}")

            idxs_pos = concept_index_pos[cname]
            idxs_pos_split = np.array([i for i in idxs_pos if i in idxs_split])

            ## Only use given classes for forced concepts
            forced_label = FORCED_CONCEPTS.get(cname, None)
            if forced_label is not None:
                logger.info(f"Only cosiderung samples from class {forced_label} for concept {cname}")
                idxs_label = np.where(dataset.metadata.label.values == forced_label)[0]
                idxs_pos_split = np.array([idx for idx in idxs_pos_split if idx in idxs_label])

            if len(idxs_pos_split) > 1000:
                idxs_pos_split = np.random.choice(idxs_pos_split, 1000, replace=False)


            ds_split_concept = dataset.get_subset_by_idxs(idxs_pos_split)
            ds_split_no_concept = dataset_wo_concept.get_subset_by_idxs(idxs_pos_split)

            dl_concept = DataLoader(ds_split_concept, batch_size=1, shuffle=False)
            dl_no_concept = DataLoader(ds_split_no_concept, batch_size=1, shuffle=False)
            
            # Register forward hook for layer of interest
            layer = config["layer_name"]
            for n, m in model.named_modules():
                if n.endswith(layer):
                    m.register_forward_hook(get_activation)

            TCAV_sens_list = []
            TCAV_pos = 0
            TCAV_neg = 0
            TCAV_pos_mean = 0
            TCAV_neg_mean = 0
            for i, (batch_concept, batch_no_concept) in enumerate(tqdm.tqdm(zip(dl_concept, dl_no_concept))):
                
                # if i >= 50:
                #     break
                x_concept, y, _ = batch_concept
                ys_all[cname].append(y)
                x_no_concept, _, _ = batch_no_concept
                
                y_hat_no_concept = model(x_no_concept.to(device))
                preds_no_concept[cname].append(y_hat_no_concept.clone().detach().cpu())
                x_no_concept_latent = activations.clone()
                model.zero_grad()

                grad_target = y

                # Compute latent representation
                with torch.enable_grad():
                    x_concept.requires_grad = True
                    y_hat_concept = model(x_concept.to(device))
                    preds[cname].append(y_hat_concept.clone().detach().cpu())
                    yc_hat = y_hat_concept[:, grad_target]

                    grad = torch.autograd.grad(outputs=yc_hat,
                                            inputs=activations,
                                            retain_graph=True,
                                            grad_outputs=torch.ones_like(yc_hat))[0]

                    grad = grad.detach().cpu()
                    model.zero_grad()
                    
                    x_concept_latent = activations.clone()

                    cav_sample = (x_concept_latent - x_no_concept_latent).detach().cpu()
                    if config["cav_dim"] == 1:
                        cav_sample = cav_sample if cav_sample.dim() == 2 else cav_sample.flatten(start_dim=2).max(2).values
                    cav_sample = cav_sample / torch.sqrt((cav_sample ** 2).sum())

                    grad = grad if grad.dim() > 2 else grad[..., None, None]
                    
                    sens_gt = (grad * cav_sample[..., None, None]).sum((0, 1))
                    sens_cav = (grad * cav[..., None, None]).sum((0, 1))
                    
                    
    

                    tcav_metrics_batch = compute_tcav_metrics_batch(grad, cav_sample)
                    
                    TCAV_pos += tcav_metrics_batch["TCAV_pos"]
                    TCAV_neg += tcav_metrics_batch["TCAV_neg"]
                    TCAV_pos_mean += tcav_metrics_batch["TCAV_pos_mean"]
                    TCAV_neg_mean += tcav_metrics_batch["TCAV_neg_mean"]

                    TCAV_sens_list.append(tcav_metrics_batch["TCAV_sensitivity"])

                with torch.enable_grad():
                    x_no_concept.requires_grad = True
                    _y_hat = model(x_no_concept.to(device))
                    yc_hat = _y_hat[:, grad_target]

                    grad_no_concept = torch.autograd.grad(outputs=yc_hat,
                                                        inputs=activations,
                                                        retain_graph=True,
                                                        grad_outputs=torch.ones_like(yc_hat))[0]

                    grad_no_concept = grad_no_concept.detach().cpu()

                    grad_no_concept = grad_no_concept if grad_no_concept.dim() > 2 else grad_no_concept[..., None, None]
                    model.zero_grad()
                    
                    sens_no_concept_gt = (grad_no_concept * cav_sample[..., None, None]).sum((0, 1))
                    sens_no_concept_cav = (grad_no_concept * cav[..., None, None]).sum((0, 1))

                    name = f"results/cav_sensitivities_neg/{config['cav_type']}/{cname}/sample_{i}.pdf"
                    store_sens_hm(ds_split_concept, sens_gt, sens_cav, sens_no_concept_gt, sens_no_concept_cav, x_concept, x_no_concept, name)
                

            tcav_metrics = aggregate_tcav_metrics(TCAV_pos, TCAV_neg, TCAV_pos_mean, TCAV_neg_mean, TCAV_sens_list)

            results[f"{split}_mean_tcav_quotient_gt_{cname}"] = tcav_metrics['mean_tcav_quotient']
            results[f"{split}_mean_quotient_gt_sderr_{cname}"] = tcav_metrics['mean_quotient_sderr']

            results[f"{split}_tcav_quotient_gt_{cname}"] = tcav_metrics['tcav_quotient']
            results[f"{split}_quotient_gt_sderr_{cname}"] = tcav_metrics['quotient_sderr']

            results[f"{split}_mean_tcav_sensitivity_gt_{cname}"] = tcav_metrics['mean_tcav_sensitivity']
            results[f"{split}_mean_tcav_sensitivity_gt_sem_{cname}"] = tcav_metrics['mean_tcav_sensitivity_sem']

            if config.get('wandb_api_key', None):
                wandb.log({**results, **config})
        print("Done.")

def store_sens_hm(ds, sens_gt, sens_cav, sens_no_concept_gt, sens_no_concept_cav, x_concept, x_no_concept, name):
    norm_val = max(sens_cav.abs().max(), sens_no_concept_cav.abs().max())
    norm_val_gt = max(sens_gt.abs().max(), sens_no_concept_gt.abs().max())
    sens_gt /= norm_val_gt
    sens_cav /= norm_val
    sens_no_concept_gt /= norm_val_gt
    sens_no_concept_cav /= norm_val
    fig, axs = plt.subplots(1, 6, figsize=(12,3))
    
    axs[0].imshow(ds.reverse_normalization(x_concept).numpy()[0].transpose(1,2,0))
    axs[1].imshow(ds.reverse_normalization(x_no_concept).numpy()[0].transpose(1,2,0))
    axs[2].imshow(imgify(sens_cav.detach().numpy(), vmin=-1., vmax=1., level=1, cmap="bwr"))
    axs[3].imshow(imgify(sens_gt.detach().numpy(), vmin=-1., vmax=1., level=1, cmap="bwr"))
    axs[4].imshow(imgify(sens_no_concept_cav.detach().numpy(), vmin=-1., vmax=1., level=1, cmap="bwr"))
    axs[5].imshow(imgify(sens_no_concept_gt.detach().numpy(), vmin=-1., vmax=1., level=1, cmap="bwr"))
    
    for i in range(6):
        axs[i].axis("off")

    axs[0].set_title("Input")
    axs[1].set_title("Input (Random c)")
    axs[2].set_title("Senitivity - pos (CAV)")
    axs[3].set_title("Senitivity - pos (GT)")
    axs[4].set_title("Senitivity - neg (CAV)")
    axs[5].set_title("Senitivity - neg (GT)")

    os.makedirs(os.path.dirname(name), exist_ok=True)
    fig.savefig(name)
    plt.close()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
