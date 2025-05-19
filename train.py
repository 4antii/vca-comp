import os
import yaml
import glob
import torch
import torchsummary
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser

from data import datasets_map
from models import models_map
from utils import Config

parser = ArgumentParser()
parser.add_argument('--config_path', type=str, help='Path to model config', required=True)
parser.add_argument('--dataset', type=str, default='alesis3630')

args = parser.parse_args()

with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

config = Config(config)

print(f"Using {args.dataset} Dataset")
dataset_config = datasets_map[args.dataset]

train_dataset = dataset_config['dataset_class'](dataset_config['train_source'], 
                                                dataset_config['train_targets'],
                                                subset=config.train_subset,
                                                half=True if config.precision == 16 else False,
                                                preload=config.preload,
                                                length=config.train_length,
                                                params_num=4)

val_dataset = dataset_config['dataset_class'](dataset_config['val_source'],
                                              dataset_config['val_targets'],
                                              preload=config.preload,
                                              half=True if config.precision == 16 else False,
                                              subset=config.val_subset,
                                              length=config.eval_length,
                                              params_num=4)

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               shuffle=config.shuffle,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                    shuffle=False,
                                    batch_size=8,
                                    num_workers=config.num_workers,
                                    pin_memory=True)

config.nparams = dataset_config['nparams']

model = models_map[config.model_type](**config.to_dict())

default_root_dir = os.path.join('experiments', args.dataset, config.exp_name)

trainer = pl.Trainer(
    max_epochs=config.max_epochs,
    precision=config.precision,
    default_root_dir=default_root_dir,
    accelerator="gpu" if torch.cuda.is_available() else "cpu")

trainer.fit(model, train_dataloader, val_dataloader)