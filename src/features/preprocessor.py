from dataclasses import dataclass
from typing import Callable
from enum import Enum

import numpy as np
import torchaudio


class Wav2FreqConverter(Enum):
    fft = 'fft'
    mel = 'mel'
    stft = 'stft'


@dataclass
class WaveProcessorConfig:
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    f_min: int = 0
    f_max: int = 20000
    pad: int = 0


class WaveProcessor:
    __converter: Callable[[np.ndarray, int], np.ndarray] = None

    def __init__(self, config):
        self.config = config
        self.__converter = self._fft

    def get_config(self) -> WaveProcessorConfig:
        return self.config

    def set_config(self, config: WaveProcessorConfig) -> None:
        self.config = config

    def _fft(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Convert waveform to Fast Fourier Transform
        """
        fft = np.fft.fft(waveform)
        return fft

    def _mel_spectrogram(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Convert waveform to mel spectrogram
        """
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max
        )(waveform)
        return mel_spectrogram

    def _stft(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Convert waveform to short time fourier transform
        """
        swft = torchaudio.transforms.Spectrogram(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            pad=self.config.pad,
            power=None
        )(waveform)
        return swft

    def set_wav2freq_converter(self, converter: Wav2FreqConverter) -> None:
        """
        Set frequency converter
        :param converter: frequency converter
        :return: None
        """
        if converter == Wav2FreqConverter.fft:
            self.__converter = self._fft
        elif converter == Wav2FreqConverter.mel:
            self.__converter = self._mel_spectrogram
        elif converter == Wav2FreqConverter.stft:
            self.__converter = self._stft
        else:
            raise ValueError(f"Invalid wav2freq converter: {converter}")

    def convert(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Convert waveform to frequency domain
        :param waveform: waveform data
        :param sr: sampling rate
        :return: frequency domain data: np.ndarray(freqs, time, intensity)
        """
        return self.__converter(waveform, sr)

    def denoise(self, spectrogram: np.ndarray, min_freq: int = 0, max_freq: int = 20000) -> np.ndarray:
        """
        Denoise spectrogram
        :return: denoised spectrogram
        """
        return spectrogram[min_freq:max_freq, :]