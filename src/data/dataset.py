from dataclasses import dataclass
from typing import List, Tuple, TypeVar

import numpy as np
import pandas as pd
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

import librosa

import os
import soundfile as sf


E = TypeVar('E')
Waveform = np.ndarray[(np.intp, np.intp), E]
Spectrogram = np.ndarray[(np.intp, np.intp), E]
SpectroData = Tuple[Spectrogram, np.ndarray | None, DataFrame | None]


class WaveDataset(Dataset):
    """
    Dataset class for loading WAV files
    """
    __files: List[str] = []
    __metadata_files: List[str] = []
    __data_dir: str = ""
    __split_ratio: float = 0.8

    def __init__(self, data_dir: str, transform=None):
        self.__data_dir = data_dir
        self.__files = os.listdir(data_dir)
        self.__transform = transform

    def __len__(self) -> int:
        return len(self.__files)

    def __getitem__(self, idx: int) -> (np.ndarray, int):
        """
        Load wav file from wav file store the wav_data and sampling rate
        :param idx: index of file to be loaded
        :return: (Waveform, sampling rate)
        """
        data_path = os.path.join(self.__data_dir, self.__files[idx])
        wav_data, sr = torchaudio.load(data_path, normalize=True)
        if self.__transform:
            wav_data = self.__transform(wav_data)
        # Split the file into x and y by split ratio
        print(wav_data.shape, sr)
        # x = wav_data[:, :int(len(wav_data) * self.__split_ratio)]
        # y = wav_data[:, int(len(wav_data) * self.__split_ratio):]
        return wav_data, sr

    def get_file_name(self, idx: int) -> str:
        return self.__files[idx]

    @staticmethod
    def save(wav_data: np.ndarray, sr: int, save_path: str, file_name: str) -> None:
        """
        Save wav to destination path with file name
        :param wav_data: waveform data in numpy form
        :param sr: sampling rate of wav file
        :param save_path: destination path
        :param file_name: name of file to be saved
        :return: None
        """
        destination_path = os.path.join(save_path, f"{file_name}.wav")
        sf.write(destination_path, wav_data, sr)




class SpectrogramDataset(Dataset):
    """
    Dataset class for loading Spectrogram files (np.array)
    """
    __files: List[str] = []
    __label_files: List[str] = []
    __metadata_files: List[str] = []
    __data_dir: str = ""
    __label_dir: str = ""
    __metadata_dir: str = ""
    __sr: int = 0

    def __init__(self, data_dir: str, transform=None, label_dir: str = None, metadata_dir: str = None):
        self.__data_dir = data_dir
        self.__label_dir = label_dir
        self.__metadata_dir = metadata_dir

        self.__files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if label_dir:
            self.__label_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        if metadata_dir:
            self.__metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.csv')]
        self.__transform = transform

    def __len__(self):
        return len(self.__files)

    def __getitem__(self, idx: int) -> SpectroData:
        """
        Load wav file from wav file store the wav_data and sampling rate
        :param idx: index of file to be loaded
        :return: (Spectrogram, Label, Metadata)
        """
        data_path = os.path.join(self.__data_dir, self.__files[idx])
        spectro: np.ndarray = np.load(data_path, mmap_mode='r')
        if self.__transform:
            spectro = self.__transform(spectro)

        if len(self.__label_files) > 0:
            label_path = os.path.join(self.__label_dir, self.__label_files[idx])
            label: np.ndarray | None = np.load(label_path, mmap_mode='r')
        else:
            label = None
        if len(self.__metadata_files) > 0:
            metadata_path = os.path.join(self.__metadata_dir, self.__metadata_files[idx])
            metadata: DataFrame | None = pd.read_csv(metadata_path)
        else:
            metadata = None
        return spectro, label, metadata

    @staticmethod
    def save(spectro_data: np.ndarray, save_path: str, file_name: str) -> None:
        """
        Save spectrogram to destination path with file name
        :param spectro_data: wav data in numpy form
        :param save_path: destination path
        :param file_name: name of file to be saved
        :return: None
        """
        destination_path = os.path.join(save_path, f"{file_name}.npy")
        np.save(destination_path, spectro_data)


@dataclass
class Datasets:
    train: Dataset
    val: Dataset
    test: Dataset


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


if __name__ == "__main__":
    dataset = WaveDataset("data/raw/musicnet/train_data")
    dl = DataLoader(dataset, batch_size=1, shuffle=True)
    for data, sr in dl:
        print(data.shape, sr)
        break
