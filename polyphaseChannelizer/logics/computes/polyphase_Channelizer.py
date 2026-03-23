import numpy as np
import pydantic
import cupy as cp
from logics.compute_base import ComputeBase
from utils.types import ArrayType

class polyphaseChannelizer(ComputeBase):
    class Config(ComputeBase.Config):
        fs_hz: pydantic.PositiveInt
        bw_hz : pydantic.PositiveInt
        filter_path: str
        use_gpu: bool = False

        def create_logical_instance(self):
            return polyphaseChannelizer(config=self)

    def initialize(self):
        self.type = cp if self.config.use_gpu else np
        taps_cpu = np.loadtxt(self.config.filter_path, dtype=np.float32)
        self.filter_taps = self.type.asarray(taps_cpu)
        self.num_filter_taps = self.filter_taps.size
        self.num_channels = int(self.config.fs_hz // self.config.bw_hz)
        self.OLA = int(self.num_filter_taps // self.num_channels)

    def compute(self, data: ArrayType) -> ArrayType:
        """
        preform the logic of polyphase channelizer. devides a spectrum with width fs to narrow channnels with
        width of bw. for the general case of no overlap between channels. (decimation factor equals num channles)
        :param 
            data (type.ndarray): The flat input signal (complex64).
            taps (type.ndarray): Filter coefficients for the prototype filter.
        :return:
            type.ndarray: A 2D matrix of shape (frames, num_channels).
        """
        data = self.type.asarray(data)
        num_samples_raw = data.size
        num_samples = (num_samples_raw // self.num_channels) * self.num_channels 

        data_mat = data.reshape(-1, self.num_channels)
        banks_mat = self.filter_taps.reshape(self.OLA, self.num_channels)

        convolution_length = data_mat.shape[0] + banks_mat.shape[0] - 1
        data_fft = self.type.fft.fft(data_mat, n=convolution_length, axis=0)
        banks_fft = self.type.fft.fft(banks_mat, n=convolution_length, axis=0)
        filtered_fft = data_fft * banks_fft
        filterd_and_decimated_mat = self.type.fft.ifft(filtered_fft, axis=0)[:data_mat.shape[0], :]
        
        inverse_DFT_result = self.type.fft.fftshift(self.type.fft.ifft(filterd_and_decimated_mat, axis=1), axes=1)
        output_signal  =  self.num_channels * inverse_DFT_result 

        return output_signal.astype(self.type.complex64)



        