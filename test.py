import os
import sys
import time
import yaml
import glob
import json
import torch
import pickle
import auraloss
import torchaudio
import numpy as np
import torchsummary
from pathlib import Path
from tqdm import tqdm
from thop import profile
import pyloudnorm as pyln
import pytorch_lightning as pl
from argparse import ArgumentParser

from data import datasets_map
from models import models_map
from utils import Config

from models.utils import causal_crop, center_crop

pl.seed_everything(42)

# fix threads
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

parser = ArgumentParser()
parser.add_argument('--config_path', type=str, help='path to model config', required=True)
parser.add_argument('--dataset', type=str, default='alesis3630')

args = parser.parse_args()

with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

config = Config(config)

print(f"Using {args.dataset} Dataset")
dataset_config = datasets_map[args.dataset]

test_dataset = dataset_config['dataset_class'](dataset_config['test_source'] if config.eval_subset == 'test' else dataset_config['val_source'], 
                                dataset_config['test_targets'] if config.eval_subset == 'test' else dataset_config['val_targets'],
                                subset=config.eval_subset,
                                half=False,
                                preload=config.preload,
                                length=config.eval_length,
                                params_num=dataset_config['nparams'])


test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                               shuffle=False,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers)

overall_results = {}

if config.eval_audo_save_dir is not None:
    if not os.path.isdir(config.eval_audo_save_dir):
        os.makedirs(config.eval_audo_save_dir)

l1   = torch.nn.L1Loss()
mse = torch.nn.MSELoss()
stft = auraloss.freq.STFTLoss()
meter = pyln.Meter(44100)

default_root_dir = Path(f'./experiments/{args.dataset}/').resolve()
models = sorted(default_root_dir.glob('*'))

for idx, model_dir in enumerate(models):

    results = {}

    model_dir = str(Path(model_dir).resolve())

    print(model_dir)

    checkpoint_path = glob.glob(os.path.join(model_dir,
                                             "lightning_logs",
                                             "version_0",
                                             "checkpoints",
                                             "*"
                                            ))[0]
    hparams_file = os.path.join(model_dir, "hparams.yaml")

    model_id = os.path.basename(model_dir)
    model_type = '_'.join(os.path.basename(model_dir).split('_')[:2])
    epoch = int(os.path.basename(checkpoint_path).split('-')[0].split('=')[-1])

    print('model_type: ', model_type)
    print('checkpoint path: ', checkpoint_path)

    if "lstm_raw" in model_type:
        model = models_map['lstm'].load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )
    elif 'gru_multiband' in model_type or 'multiband_gru' in model_type or 'lstm_multiband' in model_type or 'multiband_lstm' in model_type:
        model = models_map['multiband'].load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )
    else:
        model = models_map['tcn'].load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cuda:0"
        )

    i = torch.rand(1,1,65536)
    p = torch.rand(1,1,4)
    macs, params = profile(model, inputs=(i, p))

    print(f" {idx+1}/{len(models)} : epoch: {epoch} {os.path.basename(model_dir)}")
    print(   f"MACs: {macs/10**9:0.2f} G     Params: {params/1e3:0.2f} k")

    model.to("cpu").eval()
    i = torch.rand(1,1,65536)
    p = torch.rand(1,1,4)
    iterations = 1000

    with torch.no_grad():
        # warm‑up
        for _ in range(10):
            _ = model(i, p)

        # benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(i, p)
        end = time.perf_counter()

    avg_time = (end - start) / iterations
    audio_duration = 65536 / 44100
    rt_factor = avg_time / audio_duration

    print(f"Average inference time: {avg_time:.6f}s")
    print(f"Realtime factor: {rt_factor:.6f}")
    print(f"Inverse Realtime factor: {1 / rt_factor:.6f}")

    model.cuda()
    model.eval()

    if config.eval_precision == 16:
        model.half()

    for bidx, batch in enumerate(test_dataloader):

        sys.stdout.write(f" Evaluating {bidx}/{len(test_dataloader)}...\r")
        sys.stdout.flush()

        input, target, params = batch

        input = input.to("cuda:0")
        target = target.to("cuda:0")
        params = params.to("cuda:0")

        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(input, params)

            # crop the input and target signals
            if model.hparams.causal:
                input_crop = causal_crop(input, output.shape[-1])
                target_crop = causal_crop(target, output.shape[-1])
            else:
                input_crop = center_crop(input, output.shape[-1])
                target_crop = center_crop(target, output.shape[-1])


        for idx, (i, o, t, p) in enumerate(zip(
                                            torch.split(input_crop, 1, dim=0),
                                            torch.split(output, 1, dim=0),
                                            torch.split(target_crop, 1, dim=0),
                                            torch.split(params, 1, dim=0))):

            l1_loss = l1(o, t).cpu().numpy()
            stft_loss = stft(o, t).cpu().numpy()
            rms_loss = torch.sqrt(mse(o, t)).cpu().numpy()
            aggregate_loss = l1_loss + stft_loss 

            target_lufs = meter.integrated_loudness(t.squeeze().cpu().numpy())
            output_lufs = meter.integrated_loudness(o.squeeze().cpu().numpy())
            l1_lufs = np.abs(output_lufs - target_lufs)

            l1i_loss = (l1(i, t) - l1(o, t)).cpu().numpy()
            stfti_loss = (stft(i, t) - stft(o, t)).cpu().numpy()

            params = p.squeeze().cpu().numpy()
            params_key = f"{params[0]}-{params[1]}-{params[2]}-{params[3]}"

            if config.eval_audo_save_dir is not None:
                ofile = os.path.join(config.eval_audo_save_dir, f"{params_key}-{bidx}-output--{model_id}.wav")
                ifile = os.path.join(config.eval_audo_save_dir, f"{params_key}-{bidx}-input.wav")
                tfile = os.path.join(config.eval_audo_save_dir, f"{params_key}-{bidx}-target.wav")

                torchaudio.save(ofile, o.view(1,-1).cpu().float(), 44100)
                if not os.path.isfile(ifile):
                    torchaudio.save(ifile, i.view(1,-1).cpu().float(), 44100)
                if not os.path.isfile(tfile):
                    torchaudio.save(tfile, t.view(1,-1).cpu().float(), 44100)

            if params_key not in list(results.keys()):
                results[params_key] = {
                    "L1" : [l1_loss],
                    "L1i" : [l1i_loss],
                    "STFT" : [stft_loss],
                    "STFTi" : [stfti_loss],
                    "RMS": [rms_loss],
                    "LUFS" : [l1_lufs],
                    "Agg" : [aggregate_loss]
                }
            else:
                results[params_key]["L1"].append(l1_loss)
                results[params_key]["L1i"].append(l1i_loss)
                results[params_key]["STFT"].append(stft_loss)
                results[params_key]["STFTi"].append(stfti_loss)
                results[params_key]["RMS"].append(rms_loss)
                results[params_key]["LUFS"].append(l1_lufs)
                results[params_key]["Agg"].append(aggregate_loss)

    # store in dict
    l1_scores = []
    lufs_scores = []
    rms_scores = []
    stft_scores = []
    agg_scores = []
    print("-" * 64)
    print("Config         L1         STFT      RMS       LUFS      ")
    print("-" * 64)
    for key, val in results.items():
        l1_scores += val["L1"]
        stft_scores += val["STFT"]
        rms_scores += val["RMS"]
        lufs_scores += val["LUFS"]
        agg_scores += val["Agg"]

    print(f"Mean Error {np.mean(l1_scores):0.2e}    {np.mean(stft_scores):0.3f}    {np.mean(rms_scores):0.4f}      {np.mean(lufs_scores):0.3f}")
    overall_results[model_id] = results