import numpy as np

def generate_test_input(fs, bw, num_channels, num_samples):
    t = np.arange(num_samples) / fs
    full_signal = np.zeros(num_samples, dtype=np.complex64)
    expected_signals = np.zeros(num_channels, dtype=np.float32)
    channel_idx = np.arange(num_channels)
    f_centers = channel_idx * bw
    amplitudes = (channel_idx + 1)/num_channels
    signal_mat = amplitudes.reshape(-1,1) * np.exp(1j * 2 * np.pi * f_centers.reshape(-1,1) * t.reshape(1,-1) ) 
    signal = np.sum(signal_mat,axis=0).astype(np.complex64)
    
    return signal, amplitudes.reshape(-1)