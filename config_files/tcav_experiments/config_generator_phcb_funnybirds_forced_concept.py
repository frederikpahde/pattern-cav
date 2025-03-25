import copy
import itertools
import os
import shutil

import yaml

from utils.layer_names import LAYER_NAMES_BY_MODEL

config_dir = "config_files/tcav_experiments/funnybirds_forced_concept"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'num_epochs': 20,
    'eval_every_n_epochs': 5,
    'store_every_n_epochs': 20,
    'loss': 'cross_entropy',
    'device': 'cuda',
    'dataset_name': 'funnybirds_forced_concept',
    'dataset_name_attribute': 'funnybirds_attribute',
    'wandb_api_key': local_config['wandb_key'],
    'wandb_project_name': 'WANDB_PROJECT_NAME',
    'img_size': 224,
    'max_num_concepts': None
}


def store_local(config, config_name):
    _config = copy.deepcopy(config)
    _config['ckpt_path'] = "PATH_TO_CHECKPOINT"
    _config['batch_size'] = local_config['local_batch_size']
    _config['model_savedir'] = local_config['checkpoint_dir']
    _config['data_paths_attribute'] = [local_config['funnybirds_forced_concept_dir']]
    _config['data_paths'] = [local_config['funnybirds_forced_concept_dir']]
    _config['dir_precomputed_data'] = local_config['dir_precomputed_data']
    _config['wandb_project_name'] = f"local_{_config['wandb_project_name']}"
    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

for dataset_name_attribute in [
    "funnybirds_attribute"
]:
    base_config['dataset_name_attribute'] = dataset_name_attribute
    for model_name, layer_names in [
        # ('vgg16', LAYER_NAMES_BY_MODEL["vgg16_with_relu"]),
        ('vgg16', LAYER_NAMES_BY_MODEL["vgg16"][-2:]),
        # ('efficientnet_b0', ['last_conv']),
        # ('efficientnet_b0', LAYER_NAMES_BY_MODEL["efficientnet_b0"]),
        # ('resnet18', ['last_conv']),
        # ('resnet18', LAYER_NAMES_BY_MODEL["resnet18"]),
        # ('vit_b_16', 'correction_layer'),
    ]:
        for layer_name in layer_names:
            base_config['lr'] = 0.001 if model_name == "efficientnet_b0" else 0.005
            base_config['optimizer'] = "adam" if model_name == "efficientnet_b0" else "sgd"
            base_config['model_name'] = model_name
            base_config['layer_name'] = layer_name

            for cav_dim in [
                1,
                ]:
                base_config['cav_dim'] = cav_dim

                #PHCB Models
                phcb_model_name = "phcb_model"
                base_config['phcb_model_name'] = phcb_model_name

                cavs = [
                    "svm", "signal", 
                    # "lasso", "ridge", "logistic"
                    ]
                preprocessings = [""
                                #   , "-centered", "-max_scaled", "-centered-max_scaled"
                                  ]

                cav_types = [c + p for c, p in itertools.product(cavs, preprocessings)]

                for cav_type in cav_types:
                
                    base_config['cav_type'] = cav_type
                    config_name = f"{model_name}_{dataset_name_attribute}_{phcb_model_name}_{cav_type}-{cav_dim}d-{layer_name}"
                    store_local(base_config, config_name)

