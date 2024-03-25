from dataclasses import dataclass


@dataclass
class WaveProcessorConfig:
    sr: int = 44100
    n_mels: int = 128
    n_fft: int = 400
    hop_length: int = 200
    f_min: int = 0
    f_max: int = 20000
    pad: int = 0

    def to_dict(self):
        return self.__dict__
