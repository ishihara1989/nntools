import numpy as np
import scipy.io.wavfile as wavfile


# harvest
if __name__ == "__main__":
    sr, x = wavfile.read("../data/jvs/jvs_16k/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav")
    target_fs = 8000 # hyper params
    rate = np.exp2(int(np.log2(fs/target_fs)));
    actual_fs = fs/rate
    f0_floor = 70
    f0_ceil = 800
    cut_upper = actual_fs*0.975
    cut_lower = 50
    adjusted_f0_floor = f0_floor * 0.9
    adjusted_f0_ceil = f0_ceil * 1.1
    channels_in_octave = 20
    number_of_channels = 1 + int(np.log2(adjusted_f0_ceil / adjusted_f0_floor) * channels_in_octave)
    boundary_f0_list = adjusted_f0_floor * np.exp2((np.arange(number_of_channels) + 1.0) / channels_in_octave)
    max_cwt_delay = 1+2*int(round(2.0 * actual_fs / boundary_f0_list[0]))
    upper_rate = fs/cut_upper
    lower_rate = fs/cut_lower
    decimation_size=256
    filter_size = int(decimation_size * rate)
    window = np.hanning(filter_size)
    filt0 = window * (np.sinc((np.arange(filter_size)-filter_size/2)/upper_rate)/upper_rate - np.sinc((np.arange(filter_size)-filter_size/2)/lower_rate)/lower_rate)
    fft_size = int(np.exp2(np.ceil(np.log2(filter_size))))
    if fft_size==filter_size:
        filt = filt0
    else:
        filt = np.zeros(fft_size, dtype=np.float64)
        filt[:filter_size] = filt0
    # downsample
    filter_fft = np.fft.rfft(filt)
    f,t,spec = scipy.signal.stft(x, window='hann', nperseg=fft_size, noverlap=fft_size//2)
    spec[:,:] *= filter_fft[:, None]
    t, ds = scipy.signal.istft(spec[:fft_size//4+1,:], window='hann', nperseg=fft_size//rate, noverlap=fft_size//2//rate)
    # wavelet transform
    # zc
    # raw candidate
    # overlap
    # refine
    # fix