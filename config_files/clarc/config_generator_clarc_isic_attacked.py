import copy
import os
import shutil
import yaml

from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/clarc/isic_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'num_epochs': 10,
    'device': 'cuda',
    'dataset_name': 'isic_attacked',
    'loss': 'cross_entropy',
    'wandb_api_key': local_config['wandb_key'],
    'img_size': 224,
    'wandb_project_name': 'WANDB_PROJECT_NAME',
    "p_artifact": .01,
    'attacked_classes': ['MEL'],
    'artifact': 'artificial',
    'artifact_type': "ch_time",
    'time_format': "datetime",
    'plot_alignment': False
}


def store_local(config, config_name):
    model_name = config['model_name']
    config['ckpt_path'] = "PATH_TO_CHECKPOINT"
    config['batch_size'] = local_config['local_batch_size']
    config['data_paths'] = [local_config['isic2019_dir']]
    config['checkpoint_dir_corrected'] = local_config['checkpoint_dir_corrected']
    config['dir_precomputed_data'] = local_config['dir_precomputed_data']

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


configs_by_model_name = {
    "vgg16": (0.01, "sgd", [0.0005]),
    "resnet18": (0.01, "sgd", [0.0005]),
    "efficientnet_b0": (0.01, "adam", [0.0001])
    }

for model_name, layer_names in [
    # ('efficientnet_b0', LAYER_NAMES_BY_MODEL["efficientnet_b0"]),
    ('vgg16', LAYER_NAMES_BY_MODEL["vgg16"][-3:]),
    # ('resnet18', LAYER_NAMES_BY_MODEL["resnet18"])
    # ('vgg16', ["features.21", "features.24", "features.26", "features.28"]),
    # ('resnet18', ["last_conv"]),
    # ('efficientnet_b0', ["last_conv"])
]:
    
    p_artifact, optim_name, lrs = configs_by_model_name[model_name]
    base_config['model_name'] = model_name
    base_config['p_artifact'] = p_artifact
        
    for layer_name in layer_names:
        base_config['layer_name'] = layer_name
        
        for lr in lrs:
            base_config['lr'] = lr

            base_config['optimizer'] = optim_name
            base_config['cav_scope'] = base_config["attacked_classes"]

            config = copy.deepcopy(base_config)

            ## Vanilla
            config_vanilla = copy.deepcopy(base_config)
            method = 'Vanilla'
            config_vanilla['method'] = method
            config_vanilla['lamb'] = 0.0
            config_name = f"{model_name}_{method}_{optim_name}_lr{lr}_{layer_name}"
            store_local(config_vanilla, config_name)

            config_vanilla = copy.deepcopy(base_config)
            config_vanilla['method'] = method
            config_vanilla['lamb'] = 0.0
            config_vanilla['num_epochs'] = 0
            config_name = f"{model_name}_{method}-0epochs_{optim_name}_lr{lr}_{layer_name}"
            store_local(config_vanilla, config_name)


            ## ClArC

            for direction_mode in [
                # "lasso",
                # "logistic",
                # "ridge",
                # "svm",
                # "svm-centered",
                # "svm-centered-max_scaled",
                "signal"
            ]:
                for cav_mode in [
                    # "cavs_full",
                    "cavs_max"
                ]:
                    for method in [
                        "RRClarc",
                        # "AClarc",
                        # "PClarc"
                    ]:
                        lambs = [
                            # 1e1,5 * 1e1, 
                            # 1e2,5 * 1e2, 
                            # 1e3,5 * 1e3, 
                            # 1e4,5 * 1e4, 
                            1e5,
                            # 5 * 1e5, 
                            # 1e6, 5 * 1e6, 
                            # 1e7, 
                            # 5 * 1e7, 
                            # 1e8,5 * 1e8, 
                            # 1e9,5 * 1e9, 
                            # 1e10,5 * 1e10,
                            # 1e11,5 * 1e11
                        ] if method == "RRClarc" else [1e0]

                        for lamb in lambs:
                            
                            config_clarc = copy.deepcopy(base_config)
                            config_clarc["lamb"] = lamb
                            config_clarc["method"] = method
                            config_clarc["cav_mode"] = cav_mode
                            config_clarc["direction_mode"] = direction_mode

                            if method == "RRClarc":
                                config_clarc["criterion"] = "all_logits_random"
                                config_clarc["compute"] = "l2_mean"
                            
                            config_name = f"{model_name}_{method}_lamb{lamb:.0f}_{direction_mode}_{cav_mode}_{optim_name}_lr{lr}_{layer_name}"
                            store_local(config_clarc, config_name)