import logging
import os
import numpy as np
import tqdm
import torch
import yaml
from argparse import ArgumentParser
import wandb
from datasets import get_dataset
from models import get_fn_model_loader
from utils.cav import compute_cav, get_latent_encoding_dl
from torch.utils.data import DataLoader
from sklearn import metrics
from scipy.stats import ttest_ind

FORCED_CONCEPTS = {
    "beak::beak01.glb::yellow": 0,
    "beak::beak02.glb::yellow": 1,
    "beak::beak03.glb::yellow": 2,
    "beak::beak04.glb::yellow": 3,
    "wing::wing01.glb::red": 4,
    "wing::wing02.glb::red": 5,
    "wing::wing01.glb::green": 6,
    "wing::wing02.glb::green": 7,
    "wing::wing01.glb::blue": 8,
    "wing::wing02.glb::blue": 9,
}

def get_parser():
    parser = ArgumentParser(
        description='Train Concept Bank for Post-hoc Bottleneck Model.', )
    parser.add_argument('--config_file',
                        default="config_files/training/funnybirds_forced_concept/local/vgg16_sgd_lr0.001.yaml")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            config_name = os.path.basename(args.config_file)[:-5]
        except yaml.YAMLError as exc:
            logging.info(exc)


    config['wandb_id'] = config_name
    config['config_name'] = config_name
    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    fit_concept_bank(config)


def fit_concept_bank(config):
    dataset_name_attribute = config['dataset_name_attribute']
    dataset_name = config['dataset_name']
    data_paths_attribute = config.get('data_paths_attribute', [])
    data_paths = config.get('data_paths', [])
    model_name = config['model_name']
    ckpt_path = config['ckpt_path']
    batch_size = config['batch_size']
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.get('device', default_device)
    layer_name = config['layer_name']
    cav_type = config['cav_type']
    cav_dim = config["cav_dim"]
    max_num_concepts = config.get("max_num_concepts", None)
    model_savedir = config['model_savedir']

    ds_class = get_dataset(dataset_name)(data_paths, image_size=config["img_size"], normalize_data=True)
    ds = get_dataset(dataset_name_attribute)(data_paths_attribute, image_size=config["img_size"], normalize_data=True)
    ds.do_augmentation = False

    model = get_fn_model_loader(model_name)(ckpt_path=ckpt_path, 
                                            n_class=len(ds_class.class_names))
    
    model = model.eval().to(device)
    # max_num_concepts = 2
    concept_index_pos, concept_index_neg = ds.build_pos_neg_concept_indexes()
    concept_bank = {}
    concept_bank_N = {}
    logger.info(f"Fitting {len(concept_index_pos.keys())} CAVs")
    for i, ((attr_name, idxs_pos), (_, idxs_neg)) in tqdm.tqdm(enumerate(
        zip(concept_index_pos.items(), concept_index_neg.items()))):


        if max_num_concepts and i >= max_num_concepts:
            break

        ## Compute N CAVs for statistical testing
        N = 5
        for run_n in tqdm.tqdm(range(N)):
            rng = np.random.default_rng(run_n)
            idxs_neg_all = list(set(np.arange(len(ds))) - set(idxs_pos))

            use_subset = True
            size_subset = 667
            idxs_pos_subset = rng.choice(idxs_pos, size_subset, replace=len(idxs_pos)<size_subset) if use_subset else idxs_pos

            replace = len(idxs_neg_all) < len(idxs_pos_subset)
            idxs_neg = rng.choice(idxs_neg_all, len(idxs_pos_subset), replace=replace)

            if cav_dim == 3:
                # latent representations not precomputed
                idxs_pos_train = [i for i in idxs_pos_subset if i in ds.idxs_train]
                idxs_neg_train = [i for i in idxs_neg if i in ds.idxs_train]
                ds_attr_pos = ds.get_subset_by_idxs(idxs_pos_train)
                ds_attr_neg = ds.get_subset_by_idxs(idxs_neg_train)
                dl_pos = DataLoader(ds_attr_pos, batch_size=batch_size, num_workers=8)
                dl_neg = DataLoader(ds_attr_neg, batch_size=batch_size, num_workers=8)
                latent_encoding_pos = get_latent_encoding_dl(model, dl_pos, layer_name, device, cav_dim)
                latent_encoding_neg = get_latent_encoding_dl(model, dl_neg, layer_name, device, cav_dim)
            else:
                path = f"{config['dir_precomputed_data']}/global_relevances_and_activations/{dataset_name}/{model_name}"
                mode = "cavs_max"
                latent_encoding, sample_ids = load_latent_encoding(path, layer_name, mode, "all", ds.class_names)

                sample_ids_pos = [i for i, s in enumerate(sample_ids) if (s in idxs_pos_subset) and (s in ds.idxs_train)]
                sample_ids_neg = [i for i, s in enumerate(sample_ids) if (s in idxs_neg) and (s in ds.idxs_train)]
                
                latent_encoding_pos = latent_encoding[sample_ids_pos]
                latent_encoding_neg = latent_encoding[sample_ids_neg]
            cav = compute_cav(torch.cat([latent_encoding_pos, latent_encoding_neg]).numpy(), 
                            torch.cat([torch.ones(len(latent_encoding_pos)), torch.zeros(len(latent_encoding_neg))]).numpy(),
                            cav_type)
            
            assert not torch.isnan(cav).any(), f"CAV {i} for concept {attr_name} contains NaNs"
            if run_n == 0:
                concept_bank[attr_name] = cav.squeeze()
                concept_bank_N[attr_name] = [cav.squeeze()]
            else:
                concept_bank_N[attr_name].append(cav.squeeze())

    concept_bank_savedir = f"{model_savedir}/{dataset_name_attribute}/{config['config_name']}"
    os.makedirs(concept_bank_savedir, exist_ok=True)
    torch.save(concept_bank, f"{concept_bank_savedir}/concept_bank.pt")

    eval_concept_bank(concept_bank_N, ds, model, config,  concept_index_pos, concept_index_neg)

def load_latent_encoding(precomputed_dir, layer_name, mode, split, class_ids):
    vecs, sample_ids = [], []
    for class_id in class_ids:
        path_precomputed_activations = f"{precomputed_dir}/{layer_name}_class_{class_id}_{split}.pth"
        # print(f"reading precomputed relevances/activations from {path_precomputed_activations}")
        data = torch.load(path_precomputed_activations)
        if data['samples']:
            sample_ids += data['samples']
            vecs.append(torch.stack(data[mode], 0))
    return torch.cat(vecs, 0), sample_ids

def get_activation(module, input_, output_):
    global activations
    activations = output_
    return output_.clone()

def measure_accuracy_one_concept(dl_pos, dl_neg, model, cavs, config):
    latent_encoding_pos = get_latent_encoding_dl(model, dl_pos, config['layer_name'], config['device'], config['cav_dim'])
    latent_encoding_neg = get_latent_encoding_dl(model, dl_neg, config['layer_name'], config['device'], config['cav_dim'])

    accs = []
    aucs = []
    for cav in cavs:
        scores_pos = (latent_encoding_pos * cav[None]).sum(1)
        scores_neg = (latent_encoding_neg * cav[None]).sum(1)

        pred = np.concatenate([scores_pos.numpy(), scores_neg.numpy()])
        y = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)])
        fpr, tpr, _ = metrics.roc_curve(y, pred)
        auc = metrics.auc(fpr, tpr)
        acc = ((pred>0) == y).mean()
        aucs.append(auc)
        accs.append(acc)
    return np.array(aucs).mean(), np.array(accs).mean()
    
def measure_tcav_one_concept(dl_pos, model, cavs, config):
    TCAV_sens_list = [[]] * len(cavs)
    TCAV_pos = [0] * len(cavs)
    TCAV_neg = [0] * len(cavs)
    TCAV_pos_mean = [0] * len(cavs)
    TCAV_neg_mean = [0] * len(cavs)
    for batch in dl_pos:
        if len(batch) == 3:
            x_pos, y_pos, _ = batch
        else:
            x_pos, y_pos = batch
        with torch.enable_grad():
            x_pos.requires_grad = True
            x_pos = x_pos.to(config['device'])
            y_pos_hat = model(x_pos)
            yc_hat = torch.gather(y_pos_hat, 1, y_pos.view(-1,1).to(config['device']))
            
            grad = torch.autograd.grad(outputs=yc_hat,
                                    inputs=activations,
                                    retain_graph=True,
                                    grad_outputs=torch.ones_like(yc_hat))[0]

            grad = grad.detach().cpu()
            model.zero_grad()

        # Evaluate for N CAVs for statistical significance testing
        grad = grad if grad.dim() > 2 else grad[..., None, None]
        for i, cav in enumerate(cavs):
            cav_reshaped = cav[..., None, None] if config["cav_dim"] == 1 else cav.view(1, *grad.shape[1:])

            if config["cav_dim"] == 1:
                TCAV_pos[i] += ((grad * cav_reshaped).sum(1).flatten() > 0).sum().item()
                TCAV_neg[i] += ((grad * cav_reshaped).sum(1).flatten() < 0).sum().item()
                TCAV_pos_mean[i] += ((grad * cav_reshaped).sum(1).mean((1, 2)).flatten() > 0).sum().item()
                TCAV_neg_mean[i] += ((grad * cav_reshaped).sum(1).mean((1, 2)).flatten() < 0).sum().item()
                TCAV_sensitivity = (grad * cav_reshaped).sum(1).abs().flatten().numpy()
            else:
                TCAV_pos[i] += ((grad * cav_reshaped).flatten(1).sum(1) > 0).sum().item()
                TCAV_neg[i] += ((grad * cav_reshaped).flatten(1).sum(1) < 0).sum().item()
                TCAV_sensitivity = (grad * cav_reshaped).flatten(1).sum(1).abs().numpy()
        
            TCAV_sens_list[i].append(TCAV_sensitivity)

    ## Run t-test
    tcav_quotions_all = [num_pos / (num_pos + num_neg) for num_pos, num_neg in zip(TCAV_pos, TCAV_neg)]
    tcav_quotions_mean_all = [num_pos / (num_pos + num_neg + 1e-8) for num_pos, num_neg in zip(TCAV_pos_mean, TCAV_neg_mean)]

    random_tcav = np.ones_like(tcav_quotions_all) * .5
    _, pval_tcav = ttest_ind(tcav_quotions_all, random_tcav)
    tcav_quotient = np.array(tcav_quotions_all).mean()

    random_tcav_mean = np.ones_like(tcav_quotions_mean_all) * .5
    _, pval_tcav_mean = ttest_ind(tcav_quotions_mean_all, random_tcav_mean)
    mean_tcav_quotient = np.array(tcav_quotions_mean_all).mean()

    TCAV_sens_list = np.concatenate(TCAV_sens_list[0])
    mean_tcav_sensitivity = TCAV_sens_list.mean()
    mean_tcav_sensitivity_sem = np.std(TCAV_sens_list) / np.sqrt(len(TCAV_sens_list))

    return tcav_quotient, pval_tcav, mean_tcav_quotient, pval_tcav_mean, mean_tcav_sensitivity, mean_tcav_sensitivity_sem, TCAV_pos[0], TCAV_neg[0]


def eval_concept_bank(concept_bank_N, ds, model, config,  concept_index_pos, concept_index_neg):
    ## AUC computation
    results = {}
    all_accs = []
    all_aucs = []
    all_tcav_quotients = []
    all_tcav_mean_quotients = []
    all_tcav_sensitivity = []
    all_tcav_sensitivity_sem = []
    layer_name = config["layer_name"]
    batch_size = config["batch_size"]
    for n, m in model.named_modules():
        if n.endswith(layer_name):
            print("register hook in layer", n)
            m.register_forward_hook(get_activation)


    for split, idxs_split in [
        # ("train", ds.idxs_train),
        # ("val", ds.idxs_val),
        ("test", ds.idxs_test)
    ]:

        for cname, cavs in concept_bank_N.items():
            logger.info(f"Eval concept {cname} ({split})")
            idxs_pos, idxs_neg = concept_index_pos[cname], concept_index_neg[cname]
            
            rng = np.random.default_rng(0)
            idxs_neg_all = list(set(np.arange(len(ds))) - set(idxs_pos))
            replace = len(idxs_neg_all) < len(idxs_pos)
            idxs_neg = rng.choice(idxs_neg_all, len(idxs_pos), replace=replace)

            idxs_pos_split = np.array([i for i in idxs_pos if i in idxs_split])
            idxs_neg_split = np.array([i for i in idxs_neg if i in idxs_split])
        
            ## Only use given classes for forced concepts
            forced_label = FORCED_CONCEPTS.get(cname, None)
            if forced_label is not None:
                logger.info(f"Only cosiderung samples from class {forced_label} for concept {cname}")
                idxs_label = np.where(ds.metadata.label.values == forced_label)[0]
                idxs_pos_split = np.array([idx for idx in idxs_pos_split if idx in idxs_label])

            if len(idxs_pos_split) > 1000:
                idxs_pos_split = np.random.choice(idxs_pos_split, 1000, replace=False)
            if len(idxs_neg_split) > 1000:
                idxs_neg_split = np.random.choice(idxs_neg_split, 1000, replace=False)

            ds_attr_pos = ds.get_subset_by_idxs(idxs_pos_split)
            ds_attr_neg = ds.get_subset_by_idxs(idxs_neg_split)
            
            dl_pos = DataLoader(ds_attr_pos, batch_size=batch_size, num_workers=8)
            dl_neg = DataLoader(ds_attr_neg, batch_size=batch_size, num_workers=8)

            auc, acc = measure_accuracy_one_concept(dl_pos, dl_neg, model, cavs, config)
            all_aucs.append(auc)
            all_accs.append(acc)
            results[f"{split}_concept_bank_auc_{cname}"] = auc
            results[f"{split}_concept_bank_acc_{cname}"] = acc

            ## TCAV is computed wrt true class label -> model does not predict class labels form Derm7pt dataset
            if (not "derm7pt" in config["dataset_name_attribute"]): # and (config["cav_dim"] == 1):
                tcav_metrics = measure_tcav_one_concept(dl_pos, model, cavs, config)
                tcav_quotient, pval_tcav, mean_tcav_quotient, pval_tcav_mean, mean_tcav_sensitivity, mean_tcav_sensitivity_sem, n_pos, n_neg = tcav_metrics
                
                all_tcav_quotients.append(tcav_quotient)
                all_tcav_mean_quotients.append(mean_tcav_quotient)
                all_tcav_sensitivity.append(mean_tcav_sensitivity)
                all_tcav_sensitivity_sem.append(mean_tcav_sensitivity_sem)
                results[f"{split}_n_pos_{cname}"] = n_pos
                results[f"{split}_n_neg_{cname}"] = n_neg
                results[f"{split}_concept_bank_tcav_quotient_{cname}"] = tcav_quotient
                results[f"{split}_concept_bank_tcav_pval_{cname}"] = pval_tcav
                results[f"{split}_concept_bank_mean_tcav_quotient_{cname}"] = mean_tcav_quotient
                results[f"{split}_concept_bank_mean_tcav_pval_{cname}"] = pval_tcav_mean
                results[f"{split}_concept_bank_tcav_sensitivity_{cname}"] = mean_tcav_sensitivity
                results[f"{split}_concept_bank_tcav_sensitivity_sem_{cname}"] = mean_tcav_sensitivity_sem
                

        results[f"{split}_concept_bank_mean_auc"] = np.array(all_aucs).mean()
        results[f"{split}_concept_bank_mean_acc"] = np.array(all_accs).mean()

        if not "derm7pt" in config["dataset_name_attribute"]:
            results[f"{split}_concept_bank_tcav_quotients_all_concepts"] = np.array(all_tcav_quotients).mean()
            results[f"{split}_concept_bank_tcav_quotients_all_concepts2"] = np.abs(0.5 - np.array(all_tcav_quotients)).mean()
            results[f"{split}_concept_bank_mean_tcav_quotients_all_concepts"] = np.array(all_tcav_mean_quotients).mean()
            results[f"{split}_concept_bank_mean_tcav_quotients_all_concepts2"] = np.abs(0.5 - np.array(all_tcav_mean_quotients)).mean()
            results[f"{split}_concept_bank_mean_tcav_sensitivity"] = np.array(all_tcav_sensitivity).mean()
            results[f"{split}_concept_bank_mean_tcav_sensitivity_sem"] = np.array(all_tcav_sensitivity_sem).mean()
        if config.get('wandb_api_key', None):
            wandb.log({**results, **config})
    

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
