import os
from argparse import ArgumentParser

import yaml

from experiments.preprocessing.global_collect_relevances_and_activations import run_collect_relevances_and_activations


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file', 
                        default="config_files/clarc/isic_attacked/local/vgg16_RRClarc_lamb100000_signal_cavs_max_sgd_lr0.0005_features.28.yaml")
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

    config['config_file'] = args.config_file
    run_preprocessing(config)


def run_preprocessing(config):
    collect_relevances(config)


def collect_relevances(config):

    classes = config.get("attacked_classes", range(5))
    print(config["dataset_name"])
    ## Handle real artifacts
    if config["dataset_name"] == "isic":
        classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    elif config["dataset_name"] == "funnybirds":
        classes = range(50)
    elif config["dataset_name"] == "funnybirds_forced_concept":
        classes = range(10)

    for class_name in classes:
        split = "all" if config["dataset_name"] == "funnybirds" else 'all'
        run_collect_relevances_and_activations({**config,
                                                'class_name': class_name,
                                                'split': split})


if __name__ == "__main__":
    main()
