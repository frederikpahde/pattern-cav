import logging
import os
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from datasets import get_dataset, get_dataset_kwargs
from experiments.evaluation.evaluate_by_subset import evaluate_by_subset
from experiments.evaluation.evaluate_by_subset_attacked import evaluate_by_subset_attacked
from model_correction import get_correction_method
from models import get_fn_model_loader
from utils.artificial_artifact import get_artifact_kwargs

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_parser():
    parser = ArgumentParser(
        description='Run ClArC experiments.', )
    parser.add_argument('--config_file',
                        default=
                        "config_files/clarc/isic_attacked/local/vgg16_RRClarc_lamb100000_signal_cavs_max_sgd_lr0.0005_features.28.yaml")
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--cav_type', type=str, default=None)
    parser.add_argument('--direction_type', type=str, default=None)
    parser.add_argument('--num_gpu', default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)
    config["config_file"] = config_file
    config_name = os.path.basename(config_file)[:-5]

    method = args.method
    cav_mode = args.cav_type
    direction_mode = args.direction_type

    if method:    
        config["method"] = method
        config["cav_mode"] = cav_mode
        config["direction_mode"] = direction_mode
        config_name = f"{config_name}_{method}"

        if "clarc" in method.lower():
            config["lamb"] = 1
            config_name = f"{config_name}_{cav_mode}_{direction_mode}"
        if "rrclarc" in method.lower():
            config["criterion"] = "all_logits_random"
                        
    
    start_model_correction(config, config_name, args.num_gpu)


def start_model_correction(config, config_name, num_gpu):
    """ Starts model correction for given config file.

    Args:
        config (dict): Dictionary with config parameters for training.
        config_name (str): Name of given config
        num_gpu (int): Number of GPUs
    """

    logger.info(f"Running {config_name}")

    dataset_name = config['dataset_name']
    data_paths = config['data_paths']

    model_name = config['model_name']
    ckpt_path = config['ckpt_path']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    optimizer_name = config['optimizer']
    loss_name = config['loss']
    checkpoint_dir_corrected = config['checkpoint_dir_corrected']
    percentage_batches = config.get('percentage_batches', 1)
    lr = config['lr'] #/ 1000

    # Real artifacts
    artifacts_file = config.get('artifacts_file', None)

    # Attack Details
    attacked_classes = config.get('attacked_classes', [])
    p_artifact = config.get('p_artifact', .5)
    artifact_type = config.get('artifact_type', None)
    img_size = config.get('img_size', 224)

    limit_train_batches = config.get("limit_train_batches", None)
    method = config["method"]

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = config.get('device', default_device)
    wandb_project_name = config.get('wandb_project_name', None)
    wandb_api_key = config.get('wandb_api_key', None)

    do_wandb_logging = wandb_project_name is not None

    # Initialize WandB
    if do_wandb_logging:
        assert wandb_api_key is not None, f"'wandb_api_key' required if 'wandb_project_name' is provided ({wandb_project_name})"
        os.environ["WANDB_API_KEY"] = wandb_api_key
        logger.info(f"Initialized wand. Logging to {wandb_project_name} / {config_name}...")
        wandb_id = f"{config_name}" if config.get('unique_wandb_ids', True) else None
        logger_ = WandbLogger(project=wandb_project_name, name=f"{config_name}", id=wandb_id, config=config)

    # Load Data
    kwargs_data = {
        "p_artifact": p_artifact,
        "attacked_classes": attacked_classes,
        "artifact_type": artifact_type,
        "subset": config.get('subset_correction', None)
    } if config['artifact'] == 'artificial' else {"subset": config.get('subset_correction', None)}

    artifact_kwargs = get_artifact_kwargs(config)
    dataset_specific_kwargs = get_dataset_kwargs(config)

    if "funnybirds_ch" in dataset_name:
        data_paths = [f"{data_paths[0]}/train", f"{data_paths[0]}/test_ch"]

    dataset = get_dataset(dataset_name)(data_paths=data_paths,
                                        normalize_data=True,
                                        image_size=img_size,
                                        artifact_ids_file=artifacts_file,
                                        **kwargs_data,
                                        **artifact_kwargs,
                                        **dataset_specific_kwargs)

    # Load Model
    fn_model_loader = get_fn_model_loader(model_name)

    model = fn_model_loader(
        ckpt_path=ckpt_path,
        n_class=len(dataset.class_names)).to(device)

    # Construct correction kwargs
    kwargs_correction = {}
    if "clarc" in method.lower():
        kwargs_correction['class_names'] = dataset.class_names
        if "multi" in method.lower():
            kwargs_correction['artifact_sample_ids'] = {a: dataset.sample_ids_by_artifact[a] for a in config["cav_config"].keys()}
        else:
            kwargs_correction['artifact_sample_ids'] = dataset.sample_ids_by_artifact[config['artifact']]

        kwargs_correction['sample_ids'] = np.array([i for i in dataset.idxs_train])  # [i for i in dataset.idxs_val]
        kwargs_correction['mode'] = config["cav_mode"]

    # Define Optimizer and Loss function
    correction_method = get_correction_method(method)
    model_corrected = correction_method(model, config, **kwargs_correction)

    # Define Optimizer and Loss function
    model_corrected.set_optimizer(optimizer_name, model_corrected.parameters(), lr, ckpt_path)

    weights = None if dataset_name == "imagenet" else dataset.weights
    model_corrected.set_loss(loss_name, weights=weights)
        
    # Split data into train/val
    idxs_train = dataset.idxs_train
    idxs_val = dataset.idxs_val

    dataset_train = dataset.get_subset_by_idxs(idxs_train)
    dataset_val = dataset.get_subset_by_idxs(idxs_val)

    dataset_train.do_augmentation = True  

    logger.info(f"Number of samples: {len(dataset_train)} (train) / {len(dataset_val)} (val)")

    dl_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    dl_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    checkpoint_callback = ModelCheckpoint(monitor="valid_acc",
                                          dirpath=f"{checkpoint_dir_corrected}/{config_name}",
                                          filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
                                          auto_insert_metric_name=False,
                                          save_last=True,
                                          save_weights_only=True,
                                          mode="max")

    timer = Timer()

    class EvalBySubset(Callback):
        def on_train_start(self, trainer, pl_module):
            print("Training is starting")

        def on_train_end(self, trainer, pl_module):
            print("Training is ending")

        def on_train_epoch_start(self, trainer, pl_module):
            print("Saving checkpoint")
            if trainer.current_epoch >= 1:
                if "celeba" in dataset_name:
                    evaluate_by_subset(config)
                else:
                    evaluate_by_subset_attacked(config)

    trainer = Trainer(callbacks=[
        EvalBySubset() if config.get("eval_acc_every_epoch", False) else Callback(),
        checkpoint_callback,
        timer,
    ],
        devices=num_gpu,
        detect_anomaly=True,
        max_epochs=num_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_train_batches,
        # track_grad_norm=2,
        gradient_clip_val=1000.0 if "imagenet" in dataset_name else 100.0,
        accelerator="gpu",
        logger=logger_)

    trainer.fit(model_corrected, dl_train, dl_val)
    train_time = timer.time_elapsed("train")
    logger.info(f"Training time: {train_time:.2f} s")

    logger_.log_metrics({"train_time": train_time, "gpu_name": torch.cuda.get_device_name()})
    
    contains_nans = [n for n, m in model_corrected.named_parameters() if torch.isnan(m).any()]
    assert len(contains_nans) == 0, f"The following params contain NaN values: {contains_nans}"
    
    # Store checkpoint when no finetuning is done
    if config['num_epochs'] == 0 and dataset_name != "imagenet":
        os.makedirs(f"{checkpoint_dir_corrected}/{config_name}", exist_ok=True)
        shutil.copy(ckpt_path, f"{checkpoint_dir_corrected}/{config_name}/last.ckpt")
            

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
