import numpy as np


def fourier_transform(signal) -> np.ndarray:
    """
    Convert signal data from a wav file to fft
    :param signal: signal data
    :return:
    """
    fft = np.abs(np.fft.fft(signal))
    return fft


