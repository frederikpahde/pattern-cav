import copy
import logging
import os
from argparse import ArgumentParser

import torch
import yaml
from matplotlib import pyplot as plt
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from utils.render import vis_opaque_img_border
from torchvision.utils import make_grid
import numpy as np
from torchvision.transforms import Resize
from datasets import get_dataset, get_dataset_kwargs
from model_correction import get_correction_method
from models import get_canonizer, get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs
from zennit.image import imgify
from utils.layer_names import LAYER_NAMES_BY_MODEL

torch.random.manual_seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        default="config_files/clarc/isic_attacked/local/vgg16_RRClarc_lamb100000_signal_cavs_max_sgd_lr0.0005_features.28.yaml")
    parser.add_argument('--crp_file_dir', default="/media/pahde/Data/cav-improvements/crp-files")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(args.config_file)
    with open(args.config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            config = {}

    config['config_file'] = args.config_file
    config['crp_file_dir'] = args.crp_file_dir
    visualize_cavs_all(config)


def visualize_cavs_all(config):
    """ Create RelMax visualizations of most influential neurons CAVs computed with different approaches (filter/pattern)

    Args:
        config (dict): experiment config
    """
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)
    dataset_name = config['dataset_name'] + "_attacked" if "attacked" not in config['dataset_name'] else config['dataset_name']
    model_name = config['model_name']
    layer_name = config["layer_name"]
    crp_file_dir = config['crp_file_dir']
    batch_size = config['batch_size']
    config_name = os.path.basename(config['config_file'])[:-5]
    data_paths = config['data_paths']
    img_size = config.get('img_size', 224)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)

    config["method"] = "AClarc"
    config["lamb"] = 1.0
    config["cav_mode"] = "cavs_max"

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=config.get('p_artifact', 1.0),
                                        image_size=img_size,
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        **artifact_kwargs, **dataset_specific_kwargs)
    
    n_classes = len(dataset.class_names)
    ckpt_path = config['ckpt_path']
    method = config["method"]
    model = get_fn_model_loader(model_name=model_name)(n_class=n_classes, ckpt_path=ckpt_path).to(device)
    kwargs_correction = {}
    artifact_idxs_train = [i for i in dataset.idxs_train if i in dataset.sample_ids_by_artifact[config['artifact']]]
    kwargs_correction['class_names'] = dataset.class_names
    kwargs_correction['artifact_sample_ids'] = artifact_idxs_train
    kwargs_correction['sample_ids'] = dataset.idxs_train
    kwargs_correction['mode'] = config["cav_mode"]

    direction_modes = [
        "lasso",
        "logistic",
        "ridge",
        "svm",
        "signal", 
        ]
    layer_names_all = LAYER_NAMES_BY_MODEL[f"{model_name}_with_relu"][::-1]
    NUM_CONCEPTS = 6

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_map = {layer: cc for layer in layer_names_all}    
    attribution = CondAttribution(model)

    crp_path = f"{crp_file_dir}/{dataset_name}_{config_name}"

    do_fv_run = not os.path.isdir(f"{crp_path}/ActMax_sum_normed")
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.normalize_fn,
                            path=crp_path, cache=None)

    if do_fv_run:
        fv.run(composite, 0, len(dataset), batch_size=batch_size)

    n_refimgs = 8
    for mode in [
        "relevance",
        # "activation"
    ]:
        for layer_name in layer_names_all[:6]:
        
            config["layer_name"] = layer_name
            all_ref_imgs_pos = dict()
            all_ref_imgs_neg = dict()
            all_ref_imgs_abs = dict()

            for direction_mode in direction_modes:
                config["direction_mode"] = direction_mode
                
                correction_method = get_correction_method(method)
                model_iter = copy.deepcopy(model)
                model_corrected = correction_method(model_iter, config, **kwargs_correction)
                cav = model_corrected.cav.clone().detach().cpu().reshape(-1).numpy()
                
                top_idxs_pos = (-cav).argsort()[:NUM_CONCEPTS]
                top_idxs_neg = (cav).argsort()[:NUM_CONCEPTS]
                top_idxs_abs = (-np.abs(cav)).argsort()[:NUM_CONCEPTS]

                ref_imgs_pos = fv.get_max_reference(top_idxs_pos, layer_name, mode, (0, n_refimgs), 
                                                composite=composite, 
                                                # rf=False, 
                                                rf=True, 
                                                plot_fn=vis_opaque_img_border)

                ref_imgs_pos = {f"{k} ({cav[top_idxs_pos][i]:.3f})": v for i, (k, v) in enumerate(ref_imgs_pos.items())}
                
                ref_imgs_neg = fv.get_max_reference(top_idxs_neg, layer_name, mode, (0, n_refimgs), 
                                                composite=composite, 
                                                # rf=False, 
                                                rf=True, 
                                                plot_fn=vis_opaque_img_border)
                
                ref_imgs_neg = {f"{k} ({cav[top_idxs_neg][i]:.3f})": v for i, (k, v) in enumerate(ref_imgs_neg.items())}
                
                ref_imgs_abs = fv.get_max_reference(top_idxs_abs, layer_name, mode, (0, n_refimgs), 
                                                composite=composite, 
                                                # rf=False, 
                                                rf=True, 
                                                plot_fn=vis_opaque_img_border)
                
                cav_abs_sum = np.abs(cav).sum()
                ref_imgs_abs = {f"{k} ({100*np.abs(cav[top_idxs_abs][i])/cav_abs_sum:.1f}%)": v for i, (k, v) in enumerate(ref_imgs_abs.items())}
                
                all_ref_imgs_pos[direction_mode] = ref_imgs_pos
                all_ref_imgs_neg[direction_mode] = ref_imgs_neg
                all_ref_imgs_abs[direction_mode] = ref_imgs_abs
            
            base_dir = "cav_visualizations_crp_large"
            savepath = f"results/{base_dir}/{dataset_name}_{model_name}_{layer_name}_{mode}_pos"
            savepath = f"results/{base_dir}/{dataset_name}_{model_name}_{layer_name}_{mode}_neg"
            savepath = f"results/{base_dir}/{dataset_name}_{model_name}_{layer_name}_{mode}_abs"
            plot_cav_concepts(all_ref_imgs_abs, savepath)

def plot_cav_concepts(all_ref_imgs, savepath):
    nrows = len(all_ref_imgs)
    ncols = len(all_ref_imgs[list(all_ref_imgs.keys())[0]])
    size = 1.8

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols * 1.8, size * nrows), squeeze=False)
    
    resize = Resize((150, 150))

    for i_row, (cav_type, ref_imgs) in enumerate(all_ref_imgs.items()):
        axs[i_row][0].set_ylabel(cav_type)
        for i_col, (cid, imgs) in enumerate(ref_imgs.items()):
            
            ax = axs[i_row][i_col]

            grid = make_grid(
                [resize(torch.from_numpy(np.asarray(img)).permute((2, 0, 1))) for img in imgs],
                nrow=int(len(imgs) / 2),
                padding=0)
            
            ax.imshow(imgify(grid.detach().cpu()))
            ax.set_yticks([])
            ax.set_xticks([])
            title = f"Conv Filter: {cid}" if i_col == 0 else cid
            ax.set_title(title)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    _ = [plt.savefig(f"{savepath}.{ending}", dpi=300, bbox_inches='tight') for ending in ["png", "pdf"]]
    plt.close()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
