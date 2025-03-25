import copy
import os
import shutil

import yaml

config_dir = "config_files/training/funnybirds_forced_concept"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)

base_config = {
    'num_epochs': 50,
    'device': 'cuda',
    'eval_every_n_epochs': 1,
    'store_every_n_epochs': 100,
    'dataset_name': 'funnybirds',
    'loss': 'cross_entropy',
    'img_size': 224,
    'pretrained': True,
    'wandb_api_key': local_config['wandb_key'],
    'wandb_project_name': 'WANDB_PROJECT_NAME',
    "milestones": "30"
}

def store_local(config, config_name):
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['funnybirds_forced_concept_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for model_name in [
    'vgg16',
    # 'resnet18',
    # 'efficientnet_b0'
]:
    base_config['model_name'] = model_name
    lrs = [
        # 0.0005, 
        0.001, 
        # 0.005
        ]
    for lr in lrs:
        base_config['lr'] = lr
        optims = ["adam"] if "efficientnet" in model_name else ["sgd"]
        for optim_name in optims:
            base_config['optimizer'] = optim_name
            config_name = f"{model_name}_{optim_name}_lr{lr}"
            store_local(base_config, config_name)