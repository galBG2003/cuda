import numpy as np
import pydantic
from logics.compute_base import ComputeBase
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import loadmat


class simpleChannelizer(ComputeBase):
    class Config(ComputeBase.Config):
        fs_hz : pydantic.PositiveInt
        bw_hz : pydantic.PositiveInt
        def create_logical_instance(self):
            return simpleChannelizer(config=self)
    def __init__(self, config):
        super().__init__(config)
        data = loadmat('filter_100hz_cutoff.mat')
        self.filter_taps = data['Num'].flatten()  
        self.filter_taps = self.filter_taps / np.sum(self.filter_taps)  
 
    def compute(self,spectrum : np.ndarray) -> np.ndarray:
         num_sampels = spectrum.size
         num_Channels = int(self.config.fs_hz // self.config.bw_hz)
         num_samples_per_channel = int(num_sampels // num_Channels)
         t = np.arange(num_sampels) 
         output_spectrum = np.zeros((num_Channels,num_samples_per_channel),dtype = np.complex64)
         channel_idx = np.arange(num_Channels)
         center_freqs = channel_idx * self.config.bw_hz
         thetas = 2 * np.pi * center_freqs/self.config.fs_hz
         phase_matrix = np.exp(-1j * thetas.reshape(-1,1) * t) 
         shifted_spectrum = phase_matrix * spectrum
         filtered_signals = signal.lfilter(self.filter_taps, 1.0, shifted_spectrum, axis=1)
         output = filtered_signals[:, ::num_Channels]
         print(f"center_freqs: {center_freqs}")
         print(f"thetas: {thetas}")
         print(f"filter_taps sum: {np.sum(self.filter_taps)}")
         print(f"calc amplitudes: {np.mean(np.abs(output), axis=1)}")
         
         return output
         
          

         
              
              
    