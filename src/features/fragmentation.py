import numpy as np


class Fragmentation:
    def __init__(self):
        pass

    def fix_size_frag(self, wav_data: np.ndarray, sr: int , slice_size_sec=20) -> list:
        """separate this wav file into fix length slice by given length of second
        of each slice, slice_size_sec """
        slice_size = int(slice_size_sec * sr)
        num_slices = len(wav_data) // slice_size
        slices = []
        for i in range(num_slices):  # slice file into fixed size append into
            slice_start = i * slice_size
            slice_end = (i + 1) * slice_size
            slice_data = wav_data[slice_start:slice_end]
            slices.append(slice_data)
        return slices

