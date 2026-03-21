import numpy as np
import pydantic
from scipy.signal import resample_poly
from scipy.signal import lfilter
from logics.compute_base import ComputeBase


class polyphaseChannelizer(ComputeBase):
    class Config(ComputeBase.Config):
        fs_hz: pydantic.PositiveInt
        bw_hz : pydantic.PositiveInt
        filter_path: str

        def create_logical_instance(self):
            return polyphaseChannelizer(config=self)

    def initialize(self):
        self.filter_taps = np.loadtxt(self.config.filter_path, dtype=np.float32)
        self.num_filter_taps = self.filter_taps.size
    def compute(self, data: np.ndarray) -> np.ndarray:
        num_samples_raw = data.size
        num_channels = int(self.config.fs_hz // self.config.bw_hz)
        num_samples = (num_samples_raw // num_channels) * num_channels 
        num_samples_per_channel = int(num_samples // num_channels)
        OLA = int(self.num_filter_taps // num_channels)
        data_mat = data.reshape(-1, num_channels)
        banks_mat = self.filter_taps.reshape(OLA, num_channels)
        filterd_and_decimated_signal = [lfilter(banks_mat[:, i], 1, data_mat[:, i]) for i in range(num_channels)]
        filterd_and_decimated_mat = np.column_stack(filterd_and_decimated_signal)
        inverse_DFT_result = np.fft.fftshift(np.fft.ifft(filterd_and_decimated_mat, axis=1))
        output_signal  =  num_channels * inverse_DFT_result 

        return output_signal.astype(np.complex64)



        