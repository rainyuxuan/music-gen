import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"/src")

from data import WaveDataset, SpectrogramDataset
from features import WaveProcessor, WaveProcessorConfig
from torch.utils.data import DataLoader

wdir = os.getcwd().replace("\\", "/")
processed_dir = f"{wdir}/data/processed/musicnet"
sr = 44100 // 8
wav_dataset = WaveDataset(f"{wdir}/data/raw/musicnet/train_data", sr=sr, max_sec=30)
wpconfig = WaveProcessorConfig(sr=sr)
wp = WaveProcessor(wpconfig)

wav_loader = DataLoader(wav_dataset, batch_size=1, shuffle=False)

for b, (xs, fnames) in enumerate(wav_loader):
    for x, fname in zip(xs, fnames):
        input_spec = wp.wav2freq(x)

        SpectrogramDataset.save(input_spec, f"{processed_dir}/train_data", fname)
        SpectrogramDataset.save_metadata(
            wpconfig.to_dict(), f"{processed_dir}/train_meta", fname
        )