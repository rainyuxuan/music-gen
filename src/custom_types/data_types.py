from typing import Tuple, TypeVar, Annotated, Literal, Dict

import numpy as np

# from numpy import typing as npt
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader


# Array1d = Annotated[npt.NDArray[np.float32], Literal["N"]]
# Array2d = Annotated[npt.NDArray[np.float32], Literal["N", "N"]]
Array1d = np.ndarray[(int), np.float32]
Array2d = np.ndarray[(int, int), np.float32]


Waveform = Array1d  # time_frames: Amplitude
Spectrogram = Array2d  # bins x time_frames: Intensity
SpectroData = Tuple[Spectrogram, Spectrogram | None, Dict | None]
