import numpy as np
import pydantic

from logics.compute_base import ComputeBase


class polyphaseChannelizer(ComputeBase):
    class Config(ComputeBase.Config):
        fs_hz: pydantic.PositiveInt
        bw_hz : pydantic.PositiveInt
        filter_path: str
        device: str = "cpu"

        def create_logical_instance(self):
            if self.device == "gpu":
                from logics.computes.polyphase_Channelizer_gpu import polyphaseChannelizerGPU
                return polyphaseChannelizerGPU(config =self)
            return polyphaseChannelizer(config=self)

    def initialize(self):
        self.filter_taps = np.loadtxt(self.config.filter_path, dtype=np.float32)
        self.num_filter_taps = self.filter_taps.size
        self.num_channels = int(self.config.fs_hz // self.config.bw_hz)
        self.OLA = int(self.num_filter_taps // self.num_channels)

    def compute(self, data: np.ndarray) -> np.ndarray:
        """
        preform the logic of polyphase channelizer. devides a spectrum with width fs to narrow channnels with
        width of bw. for the general case of no overlap between channels. (decimation factor equals num channles)
        :param 
            data (np.ndarray): The flat input signal (complex64).
            taps (np.ndarray): Filter coefficients for the prototype filter.
            num_channels (int): Number of frequency channels (M).
            OLA (int): num of filter coefficients per channel
        :return:
        """
        num_samples_raw = data.size
        num_samples = (num_samples_raw // self.num_channels) * self.num_channels 
        num_samples_per_channel = int(num_samples // self.num_channels)
        data_mat = data.reshape(-1, self.num_channels)
        banks_mat = self.filter_taps.reshape(self.OLA, self.num_channels)
        convolution_length = data_mat.shape[0] + banks_mat.shape[0] - 1
        data_fft = np.fft.fft(data_mat, n=convolution_length, axis=0)
        banks_fft = np.fft.fft(banks_mat, n=convolution_length, axis=0)
        filtered_fft = data_fft * banks_fft
        filterd_and_decimated_mat = np.fft.ifft(filtered_fft, axis=0)[:data_mat.shape[0], :]
        inverse_DFT_result = np.fft.fftshift(np.fft.ifft(filterd_and_decimated_mat, axis=1), axes=1)
        output_signal  =  self.num_channels * inverse_DFT_result 

        return output_signal.astype(np.complex64)



        