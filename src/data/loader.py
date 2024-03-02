import librosa
import numpy as np
import torch


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self) -> torch.DataLoader:
        return None

    def save(self) -> None:
        pass


class WaveLoader(DataLoader):
    def __init__(self, data_path: str, sr: int = 16000, duration: float = 4.0):
        super().__init__(data_path)
        self.sr = sr
        self.duration = duration

    def load(self) -> torch.DataLoader:
        return None

    def save(self) -> None:
        pass


class SpectrogramLoader(DataLoader):
    def __init__(self, data_path: str, n_fft: int = 400, hop_length: int = 160, n_mels: int = 128):
        super().__init__(data_path)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def load(self) -> torch.DataLoader:
        return None

    def save(self) -> None:
        pass

