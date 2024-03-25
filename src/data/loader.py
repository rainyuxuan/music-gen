import librosa
import numpy as np
import torch
import os
import soundfile as sf


class DataLoader:
    def __init__(self):
        """
        initializing a Dataloader with the path of the wav file
        """
        return None

    def load(self, data_path):
        """
        Load wav or np file from data_path
        :param data_path: path of data to be loaded
        :return:
        """
        pass

    def save(self, save_path: str, file_name: str) -> None:
        """
        Save wav or fft to destination path with file name
        :param save_path: destination path
        :param file_name: name of file to be saved
        :return: None
        """
        pass


class WaveLoader(DataLoader):
    def __init__(self):
        super().__init__()

    def load(self, data_path) -> (np.ndarray, int):
        """
        Load wav file from wav file store the wav_data and sampling rate
        :param data_path: path to load file
        :return: numpy signal and sampling rate
        """
        wav_data, sr = librosa.load(data_path, sr=None)
        return wav_data, sr

    def save(self, wav_data: np.ndarray, sr: int, save_path, file_name) -> None:
        """
        Save wav to destination path with file name
        :param wav_data: wav data in numpy form
        :param sr: sampling rate of wav file
        :param save_path: destination path
        :param file_name: name of file to be saved
        :return: None
        """
        destination_path = os.path.join(save_path, f"{file_name}.wav")
        sf.write(destination_path, wav_data, sr)
        return None


class SpectrogramLoader(DataLoader):
    def __init__(self):
        super().__init__()

    def load(self, data_path: str) -> np.ndarray:
        """
        Load fft file
        :param data_path: path to load fft data
        :return: numpy array of fft file
        """
        fft = np.load(data_path)
        return fft

    def save(self, fft: np.ndarray, save_path: str, file_name: str) -> None:
        """
        Save all fixed size segments in fourier format to save_path
        :param fft: numpy array of fft
        :param save_path: dictionary to save fft data
        :param file_name: name of file to be saved
        :return: None
        """
        destination_path = os.path.join(save_path, f"{file_name}.npy")
        np.save(destination_path, fft)
        return None


#
