# This program is used to process wav files into spectrogram
# and save it locally for better performance

import os
import sys

module_path = os.path.abspath(os.path.join('.'))+"/src"
if module_path not in sys.path:
    sys.path.append(module_path)

from data import WaveDataset, SpectrogramDataset
from features import WaveProcessor, WaveProcessorConfig
from torch.utils.data import DataLoader

wdir = os.getcwd().replace("\\", "/")
processed_dir = f"{wdir}/data/processed/musicnet"
sr = 44100 // 8
test_wav_dataset = WaveDataset(f"{wdir}/data/raw/musicnet/test_data", sr=sr, max_sec=30)
train_wav_dataset = WaveDataset(f"{wdir}/data/raw/musicnet/train_data", sr=sr, max_sec=30)
wpconfig = WaveProcessorConfig(sr=sr)
wp = WaveProcessor(wpconfig)

test_wav_loader = DataLoader(test_wav_dataset, batch_size=1, shuffle=False)
train_wav_loader = DataLoader(train_wav_dataset, batch_size=1, shuffle=False)

for b, (xs, fnames) in enumerate(test_wav_loader):
    for x, fname in zip(xs, fnames):
        input_spec = wp.wav2freq(x)

        SpectrogramDataset.save(input_spec, f"{processed_dir}/test_data", fname)
        SpectrogramDataset.save_metadata(
            wpconfig.to_dict(), f"{processed_dir}/test_meta", fname
        )

for b, (xs, fnames) in enumerate(train_wav_loader):
    for x, fname in zip(xs, fnames):
        input_spec = wp.wav2freq(x)

        SpectrogramDataset.save(input_spec, f"{processed_dir}/train_data", fname)
        SpectrogramDataset.save_metadata(
            wpconfig.to_dict(), f"{processed_dir}/train_meta", fname
        )