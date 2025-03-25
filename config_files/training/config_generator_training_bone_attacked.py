import copy
import os
import shutil

import yaml

config_dir = "config_files/training/bone_attacked"

if os.path.isdir(config_dir):
    shutil.rmtree(config_dir)

os.makedirs(f"{config_dir}/local", exist_ok=True)

with open("config_files/local_config.yaml", "r") as stream:
    local_config = yaml.safe_load(stream)


base_config = {
    'num_epochs': 100,
    'device': 'cuda',
    'eval_every_n_epochs': 5,
    'store_every_n_epochs': 100,
    'dataset_name': 'bone_attacked',
    'loss': 'cross_entropy',
    'img_size': 224,
    'pretrained': True,
    'wandb_api_key': local_config['wandb_key'],
    'wandb_project_name': 'WANDB_PROJECT_NAME',
    'attacked_classes': [2],
    'p_artifact': 0.2,
    'milestones': "50,80",
    'artifact_type': "white_color"

}

def store_local(config, config_name):
    config['batch_size'] = local_config['local_batch_size']
    config['model_savedir'] = local_config['checkpoint_dir']
    config['data_paths'] = [local_config['bone_dir']]

    with open(f"{config_dir}/local/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


for model_name in [
    'vgg16',
    'resnet18',
    'efficientnet_b0'
]:
    base_config['model_name'] = model_name
    lrs = [0.001, 0.005, 0.0005] if model_name in ["efficientnet_b0"] else [0.005, 0.001]
    for p_backdoor in [0, .001, .005]:
        base_config['p_backdoor'] = p_backdoor
        for lr in lrs:
            base_config['lr'] = lr
            optims = ["adam"] if "efficientnet" in model_name else ["sgd"]
            for optim_name in optims:
                base_config['optimizer'] = optim_name
                config_name = f"{model_name}_p_backdoor{p_backdoor}_{optim_name}_lr{lr}"
                store_local(base_config, config_name)
