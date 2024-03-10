from visualization import visualization
from data import loader
from features import fft
from features import fragmentation

loader_wav = loader.WaveLoader()
data_path = "/Users/xunuo/Desktop/a2a-music-gen/data/raw/musicnet/musicnet/test_data/1759.wav"
data, sr = loader_wav.load(data_path)
print(f"sampling rate is {sr}")
print(f"shape of data is {data.shape}")
frag = fragmentation.Fragmentation()
slices = frag.fix_size_frag(data, sr, slice_size_sec=3)


# test save all wav file
count = 1
for slice in slices:
    loader_wav.save(slice, sr, "/Users/xunuo/Desktop/a2a-music-gen/fragmentation_test", str(count))
    count += 1


# test fft transform and visualize
slice1_fft = fft.fourier_transform(slices[0])

# test visualization
visualization.visual_fft(slice1_fft, sr)
