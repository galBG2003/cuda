import numpy as np
import pydantic
import cupy as cp

from logics.compute_base import ComputeBase
from utils.types import ArrayType


class polyphaseChannelizerOverlap(ComputeBase):
    class Config(ComputeBase.Config):
        fs_hz: pydantic.PositiveInt
        bw_hz: pydantic.PositiveInt
        overlap_factor: pydantic.PositiveFloat
        filter_path: str
        use_gpu: bool = False

        def create_logical_instance(self):
            return polyphaseChannelizerOverlap(config=self)

    def initialize(self):
        self.type = cp if self.config.use_gpu else np

        taps_cpu = np.loadtxt(self.config.filter_path, dtype=np.float32)
        self.filter_taps = self.type.asarray(taps_cpu)

        self.channel_spacing = self.config.bw_hz * (1.0 - self.config.overlap_factor)
        self.num_channels = int(self.config.fs_hz / self.channel_spacing)
        self.decimation_factor = int(self.config.fs_hz // self.config.bw_hz)

        if self.filter_taps.size % self.num_channels != 0:
            raise ValueError(f"Filter length ({self.filter_taps.size}) must be divisible by "
            f"the number of channels ({self.num_channels})"
        )        

        self.OLA = self.filter_taps.size // self.num_channels
        self.banks_mat = self.filter_taps.reshape(self.OLA, self.num_channels)
        self.past_block_data = self.type.zeros(self.filter_taps.size - 1, dtype=self.type.complex64)
        self.frames_processed = 0
    
    def polyphaseDecomposition(self,num_output_frames: int,continued_data:ArrayType,time_idx: ArrayType):
        OLA_idx = self.type.arange(self.OLA).reshape(1, self.OLA, 1)
        channel_idx = self.type.arange(self.num_channels).reshape(1, 1, self.num_channels)
        n_idx_time = time_idx.reshape(num_output_frames, 1, 1)

        sample_idx = n_idx_time - OLA_idx * self.num_channels - channel_idx

        return continued_data[sample_idx]
    
    def applyPolyphaseFilterbank(self,polyphase_tensor: ArrayType, num_output_frames: int):
        weighted_sum = self.type.sum(polyphase_tensor * self.banks_mat[None, :, :], axis=1)

        ifft_out = self.type.fft.ifft(weighted_sum, axis=1)

        n_frame = self.type.arange(num_output_frames).reshape(-1, 1) + self.frames_processed
        k_idx = self.type.arange(self.num_channels).reshape(1, -1)

        phase_correction = self.type.exp(-1j * 2 * np.pi * (self.decimation_factor / self.num_channels) * k_idx * n_frame)

        return ifft_out * phase_correction * self.num_channels

    
    def compute(self, data: ArrayType) -> ArrayType:
        data = self.type.asarray(data, dtype=self.type.complex64)    
        continued_data = self.type.concatenate([self.past_block_data, data])

        num_filter_taps = self.filter_taps.size
        first_time_index = num_filter_taps - 1

        if continued_data.size <= first_time_index:
            self.past_block_data = continued_data.copy()
            return self.type.empty((0, self.num_channels), dtype=self.type.complex64)

        num_output_frames = ((continued_data.size - 1) - first_time_index) // self.decimation_factor + 1
        if num_output_frames <= 0:
            self.past_block_data = continued_data[-(num_filter_taps - 1):].copy()
            return self.type.empty((0, self.num_channels), dtype=self.type.complex64)

        time_idx = first_time_index + self.type.arange(num_output_frames) * self.decimation_factor

        polyphase_tensor = self.polyphaseDecomposition(num_output_frames,continued_data,time_idx)

        weighted_sum = self.type.sum(polyphase_tensor * self.banks_mat[None, :, :], axis=1)

        output = self.applyPolyphaseFilterbank(polyphase_tensor,num_output_frames)

        self.past_block_data = continued_data[-(num_filter_taps - 1):].copy()
        self.frames_processed += num_output_frames

        return output.astype(self.type.complex64)