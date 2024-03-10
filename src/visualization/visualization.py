import numpy as np
import matplotlib.pyplot as plt


def visual_fft(fft: np.ndarray, sr: int) -> None:
    """
    Visualizing magnitude spectrum of fft
    :param fft: fft of a part of wav file
    :param sr: sampling rate of the fft
    :return:None
    """
    plt.figure(figsize=(10, 10))
    frequency = np.linspace(0, sr, len(fft))
    plt.plot(frequency, fft)
    plt.xlabel("Frequency (Hz)")
    plt.title("Visualization of FFT")
    plt.show()