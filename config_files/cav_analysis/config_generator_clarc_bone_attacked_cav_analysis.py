import copy
import os
import shutil
import yaml
import itertools

from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/cav_analysis/bone_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'num_epochs': 10,
    'device': 'cuda',
    'dataset_name': 'bone_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': local_config['wandb_key'],
    'img_size': 224,
    'wandb_project_name': 'WANDB_PROJECT_NAME',
    "p_artifact": .2,
    'attacked_classes': [2],
    'artifact': 'artificial',
    'plot_alignment': False
}


def store_local(config, config_name):
    model_name = config['model_name']
    config['ckpt_path'] = f"PATH_TO_CKPT"
    config['batch_size'] = local_config['local_batch_size']
    config['data_paths'] = [local_config['bone_dir']]
    config['checkpoint_dir_corrected'] = local_config['checkpoint_dir_corrected']
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name, layer_names in [
    ('efficientnet_b0', LAYER_NAMES_BY_MODEL["efficientnet_b0"]),
    ('vgg16', LAYER_NAMES_BY_MODEL["vgg16"]),
    ('resnet18_a1', LAYER_NAMES_BY_MODEL["resnet18"])
    # ('vgg16', ["features.28"]),
    # ('resnet18', ["last_conv"]),
    # ('efficientnet_b0', ["last_conv"])
]:
    base_config['model_name'] = model_name
    for artifact_type in [
            "white_color",
        ]:
    
        base_config["artifact_type"] = artifact_type
        
        for layer_name in layer_names:
            base_config['layer_name'] = layer_name
            lrs = [0.0001] if "efficientnet_b0" else [0.0005] ## LR after 1 scheduler update
            for lr in lrs:
                base_config['lr'] = lr
                optim_name = "adam" if model_name == "efficientnet_b0" else "sgd"

                base_config['optimizer'] = optim_name
                base_config['cav_scope'] = base_config["attacked_classes"]

                config = copy.deepcopy(base_config)

                ## ClArC

                cavs = ["svm", "signal", "lasso", "ridge", "logistic"]
                preprocessings = ["", "-centered", "-max_scaled", "-centered-max_scaled"]

                direction_modes = [c + p for c, p in itertools.product(cavs, preprocessings)]
                for direction_mode in direction_modes:
                    cav_mode  ="cavs_max"
                    method = "AClarc"
                    lamb = 1e0
                    
                                
                    config_clarc = copy.deepcopy(base_config)
                    config_clarc["lamb"] = lamb
                    config_clarc["method"] = method
                    config_clarc["cav_mode"] = cav_mode
                    config_clarc["direction_mode"] = direction_mode

                    config_name = f"{model_name}_{artifact_type}_{method}_lamb{lamb:.0f}_{direction_mode}_{cav_mode}_{optim_name}_lr{lr}_{layer_name}"
                    store_local(config_clarc, config_name)