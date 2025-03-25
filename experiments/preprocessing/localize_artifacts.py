import os
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from PIL import Image
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import get_dataset, get_dataset_kwargs
from model_correction import get_correction_method
from models import get_canonizer, get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--split", default="all")
    parser.add_argument("--save_localization", default=True, type=bool)
    parser.add_argument("--save_examples", default=True, type=bool)
    parser.add_argument('--cav_type', type=str, default=None)
    parser.add_argument('--direction_mode', type=str, default=None)
    parser.add_argument('--config_file',
                        default="config_files/real_artifacts_clarc/isic_old_model/local/vgg16_band_aid_AClarc_lamb1_svm_cavs_max_sgd_lr0.001_features.7.yaml")
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

    if args.cav_type:
        config["cav_mode"] = args.cav_type
        config["direction_mode"] = args.direction_mode

    localize_artifacts(config,
                       split=args.split,
                       save_examples=args.save_examples,
                       save_localization=args.save_localization)


def localize_artifacts(config: dict,
                       split: str,
                       save_examples: bool,
                       save_localization: bool):
    """Spatially localize artifacts in input samples.

    Args:
        config (dict): experiment config
        split (str): data split to use
        mode (str): CAV mode
        neurons (List): List of neurons to consider (all if None)
        save_examples (bool): Store example images
        save_localization (bool): Store localization heatmaps
        cav_type (bool): Cav optimizer
    """

    dataset_name = config['dataset_name']
    model_name = config['model_name']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_paths = config['data_paths']
    img_size = config.get('img_size', 224)
    cav_mode = config.get("cav_mode", "cavs_max")
    direction_mode = config["direction_mode"]
    results_dir = config.get('dir_precomputed_data', 'results')
    layer_name = config['layer_name']
    # Real artifacts
    artifacts_file = config.get('artifacts_file', None)
    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)
    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        p_artifact=config.get('p_artifact', 1.0),
                                        artifact_ids_file=artifacts_file,
                                        # p_artifact=1.0,
                                        image_size=img_size,
                                        artifact_type=config.get('artifact_type', None),
                                        attacked_classes=config.get('attacked_classes', []),
                                        **artifact_kwargs, **dataset_specific_kwargs)

    assert config['artifact'] in dataset.sample_ids_by_artifact.keys(), f"Artifact {config['artifact']} unknown."
    
    artifact_ids = dataset.sample_ids_by_artifact[config['artifact']]

    print(f"Chose {len(artifact_ids)} target samples.")

    model = get_fn_model_loader(model_name=model_name)(n_class=len(dataset.class_names),
                                                       ckpt_path=config['ckpt_path'])
    model = model.to(device)
    model.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    attribution = CondAttribution(model)

    img_to_plt = lambda x: dataset.reverse_normalization(x.detach().cpu()).permute((1, 2, 0)).int().numpy()
    hm_to_plt = lambda x: x.detach().cpu().numpy()

    # Construct correction kwargs
    kwargs_correction = {}
    artifact_idxs_train = [i for i in dataset.idxs_train if i in dataset.sample_ids_by_artifact[config['artifact']]]
    kwargs_correction['class_names'] = dataset.class_names
    kwargs_correction['artifact_sample_ids'] = artifact_idxs_train
    kwargs_correction['sample_ids'] = dataset.idxs_train
    kwargs_correction['mode'] = config["cav_mode"]
    correction_method = get_correction_method("Clarc")
    model_corrected = correction_method(model, config, **kwargs_correction)
    w = model_corrected.cav.clone()[..., None, None].to(device)

    samples = [dataset[i] for i in artifact_ids]
    data_sample = torch.stack([s[0] for s in samples]).to(device).requires_grad_()
    target = [s[1] for s in samples]
    conditions = [{"y": t.item() if isinstance(t, torch.Tensor) else t} for t in target]

    batch_size = config["batch_size"]
    num_batches = int(np.ceil(len(data_sample) / batch_size))

    heatmaps = []
    heatmaps_clamped = []
    inp_imgs = []

    for b in tqdm(range(num_batches)):
        data = data_sample[batch_size * b: batch_size * (b + 1)]
        attr = attribution(data,
                           conditions[batch_size * b: batch_size * (b + 1)],
                           composite, record_layer=[layer_name])
        act = attr.activations[layer_name]

        inp_imgs.extend([img_to_plt(s.detach().cpu()) for s in data])

        attr = attribution(data, [{}], composite, start_layer=layer_name, init_rel=act.clamp(min=0) * w)
        heatmaps.extend([hm_to_plt(h.detach().cpu()) for h in attr.heatmap])
        heatmaps_clamped.extend([hm_to_plt(h.detach().cpu().clamp(min=0)) for h in attr.heatmap])


    if save_examples:
        savepath = f"results/cav_heatmaps/{dataset_name}/{model_name}/{config['artifact']}_{layer_name}_{cav_mode}_{direction_mode}.png"
        plot_example_figure(inp_imgs, heatmaps, artifact_ids, savepath)
        savepath = f"results/localization/{dataset_name}/{model_name}/{config['artifact']}_{layer_name}_{cav_mode}_{direction_mode}.png"
        plot_example_figure(inp_imgs, heatmaps_clamped, artifact_ids, savepath)

    if save_localization:
        savepath = f"{results_dir}/localized_artifacts/{config['dataset_name']}/{config['layer_name']}-{config['direction_mode']}/{config['artifact']}"
        save_all_localizations(heatmaps, artifact_ids, savepath)



def plot_example_figure(inp_imgs, heatmaps, artifact_ids, savepath):
    num_imgs = min(len(inp_imgs), 72) * 2
    grid = int(np.ceil(np.sqrt(num_imgs) / 2) * 2)

    fig, axs_ = plt.subplots(grid, grid, dpi=150, figsize=(grid * 1.2, grid * 1.2))

    for j, axs in enumerate(axs_):
        ind = int(j * grid / 2)
        for i, ax in enumerate(axs[::2]):
            if len(inp_imgs) > ind + i:
                ax.imshow(inp_imgs[ind + i])
                ax.set_xlabel(f"sample {int(artifact_ids[ind + i])}", labelpad=1)
            ax.set_xticks([])
            ax.set_yticks([])

        for i, ax in enumerate(axs[1::2]):
            if len(inp_imgs) > ind + i:
                max = np.abs(heatmaps[ind + i]).max()
                ax.imshow(heatmaps[ind + i], cmap="bwr", vmin=-max, vmax=max)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"artifact", labelpad=1)

    plt.tight_layout(h_pad=0.1, w_pad=0.0)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.show()

def save_all_localizations(heatmaps, artifact_ids, savepath):
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath, exist_ok=True)
    for i in range(len(heatmaps)):
        sample_id = int(artifact_ids[i])
        heatmap = heatmaps[i]
        heatmap[heatmap < 0] = 0
        heatmap = heatmap / heatmap.max() * 255
        im = Image.fromarray(heatmap).convert("L")
        im.save(f"{savepath}/{sample_id}.png")


if __name__ == "__main__":
    main()
