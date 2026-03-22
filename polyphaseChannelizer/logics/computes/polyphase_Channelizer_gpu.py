import cupy as cp
import numpy as np
import pydantic
from logics.compute_base import ComputeBase
from logics.computes.polyphase_Channelizer import polyphaseChannelizer

class polyphaseChannelizerGPU(ComputeBase):
    class Config(polyphaseChannelizer.Config): 
        def create_logical_instance(self):
            return polyphaseChannelizerGPU(config=self)

    def initialize(self):
        taps = np.loadtxt(self.config.filter_path, dtype=np.float32)
        self.filter_taps_gpu = cp.asarray(taps)
        self.filter_taps = taps  

    def compute(self, data: np.ndarray) -> np.ndarray:
        num_channels = int(self.config.fs_hz // self.config.bw_hz)
        num_samples = (data.size // num_channels) * num_channels
        OLA = self.filter_taps_gpu.size // num_channels

        data_gpu = cp.asarray(data[:num_samples])
        data_mat = data_gpu.reshape(-1, num_channels)
        banks_mat = self.filter_taps_gpu.reshape(OLA, num_channels)

        # FFT-based convolution (no loop needed)
        N = data_mat.shape[0] + banks_mat.shape[0] - 1  # full convolution length

        # FFT both matrices along axis 0
        data_fft = cp.fft.fft(data_mat, n=N, axis=0)    # shape (N, num_channels)
        banks_fft = cp.fft.fft(banks_mat, n=N, axis=0)  # shape (N, num_channels)

        # multiply element-wise (convolution in frequency domain)
        filtered_fft = data_fft * banks_fft              # shape (N, num_channels)

        # IFFT back to time domain
        filtered = cp.fft.ifft(filtered_fft, axis=0)[:data_mat.shape[0]]  # trim to original length
        output = cp.fft.ifft(filtered, axis=1)
        # move back to CPU
        return cp.asnumpy(output).astype(np.complex64)