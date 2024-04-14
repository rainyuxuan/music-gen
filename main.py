import os
import sys

module_path = os.path.abspath(os.path.join('.'))+"/src"
if module_path not in sys.path:
    sys.path.append(module_path)

from models import *
from data import SpectrogramDataset, WaveDataset
from features import WaveProcessor, WaveProcessorConfig

import torch
from torch.utils.data import DataLoader

wdir = os.getcwd().replace("\\", "/")
processed_dir = f"{wdir}/data/processed/musicnet/test_data"
output_dir = f"{wdir}/data/result"
model_dir = f"{wdir}/models"

sr = 44100 // 8
wpconfig = WaveProcessorConfig(sr=sr)
wp = WaveProcessor(wpconfig)

model = torch.load(model_dir+"/CNN.pt")

spec_dataset = SpectrogramDataset(
    processed_dir, split_ratio=0.8
)

spec_dataloader = DataLoader(spec_dataset)
with torch.no_grad():
    model.eval()

    for data, _ , fname in spec_dataloader:
        if torch.cuda.is_available():
            data = data.cuda()
        
        out = model.generate(data, 166)

        out = torch.concat((data, out), dim=2).cpu()

        wave = wp.freq2wav(out[0])

        WaveDataset.save(wave, sr, output_dir, fname)