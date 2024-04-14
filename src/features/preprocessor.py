from dataclasses import dataclass
from typing import Callable

from custom_types import Spectrogram, Waveform
from enum import Enum

import numpy as np
import torchaudio
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from torchaudio import functional as F

from custom_types.preprocessor import WaveProcessorConfig
from data import WaveDataset


class Wav2FreqConverter(Enum):
    fft = "fft"
    mel = "mel"
    stft = "stft"


class WaveProcessor:
    __config: WaveProcessorConfig = None
    __converter: Callable[[Waveform], np.ndarray] = None

    def __init__(self, config: WaveProcessorConfig):
        self.__config = config
        self.__converter = self._stft

    def get_config(self) -> WaveProcessorConfig:
        return self.__config

    def set_config(self, config: WaveProcessorConfig) -> None:
        self.__config = config

    def _fft(self, waveform: Waveform) -> Spectrogram:
        """
        Convert waveform to Fast Fourier Transform
        """
        fft = np.fft.fft(waveform)
        return fft

    def _mel_spectrogram(self, waveform: Waveform) -> Spectrogram:
        """
        Convert waveform to mel spectrogram
        """
        # if waveform.ndim == 1:
        #     waveform = waveform.reshape(1, -1)
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.__config.sr,
            n_mels=self.__config.n_mels,
            n_fft=self.__config.n_fft,
            hop_length=self.__config.hop_length,
            f_min=self.__config.f_min,
            f_max=self.__config.f_max,
        )(waveform)
        return mel_spectrogram

    def _stft(self, waveform: Waveform) -> Spectrogram:
        """
        Convert waveform to short time fourier transform
        """
        swft = T.Spectrogram(
            n_fft=self.__config.n_fft,
            hop_length=self.__config.hop_length,
            pad=self.__config.pad,
            power=2.0,
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

    def wav2freq(self, waveform: Waveform) -> np.ndarray:
        """
        Convert waveform to frequency domain
        :param waveform: waveform data
        :param sr: sampling rate
        :return: frequency domain data: np.ndarray(freqs, time, intensity)
        """
        return self.__converter(waveform)

    def denoise(
        self, spectrogram: Spectrogram, min_freq: int = 0, max_freq: int = 20000
    ) -> Spectrogram:
        """
        Denoise spectrogram
        :return: denoised spectrogram
        """
        # TODO: Implement denoising
        return spectrogram[min_freq:max_freq, :]

    def freq2wav(self, spectrogram: Spectrogram) -> Waveform:
        """
        Convert frequency domain to waveform
        :param spectrogram: frequency domain data
        :return: waveform data
        """
        if self.__converter == self._mel_spectrogram:
            spectrogram = T.InverseMelScale(n_stft=self.__config.n_fft)(spectrogram)
        return T.GriffinLim(n_fft=self.__config.n_fft, hop_length=self.__config.hop_length, power=2)(
            spectrogram
        )


if __name__ == "__main__":
    sr = 100
    dataset = WaveDataset("data/raw/musicnet/musicnet/train_data", sr=sr)
    dl = DataLoader(dataset, batch_size=2, shuffle=False)
    for i, (x, y, fname) in enumerate(dl):
        print(fname, x.shape, y.shape, sr)  # (samples, frames), (samples)
        print(dataset.get_file_name(i))

        config = WaveProcessorConfig(sr=sr)
        wave_processor = WaveProcessor(config)

        spectro = wave_processor.wav2freq(x[0])
        print(spectro.shape)
        wave_processor.denoise(spectro)
        print(spectro.shape)
        break
