import numpy as np
import pydantic
import cupy as cp
from logics.compute_base import ComputeBase
from utils.types import ArrayType

class PolyphaseChannelizerOverlap(ComputeBase):
    class Config(ComputeBase.Config):
        fs_hz: pydantic.PositiveInt
        bw_hz: pydantic.PositiveInt
        overlap_factor: pydantic.PositiveFloat
        filter_path: str
        use_gpu: bool = False

        def create_logical_instance(self):
            return PolyphaseChannelizerOverlap(config=self)

    def initialize(self):
        self.type = cp if self.config.use_gpu else np
        self.xp_stride = cp.lib.stride_tricks if self.config.use_gpu else np.lib.stride_tricks

        taps_cpu = np.fromfile(self.config.filter_path, dtype=np.float32)
        self.filter_taps = self.type.asarray(taps_cpu)
        
        self.channel_spacing = self.config.bw_hz * (1.0 - self.config.overlap_factor)
        self.num_channels = int(self.config.fs_hz // self.channel_spacing) 
        self.decimation_factor = int(self.config.fs_hz // self.config.bw_hz)

        if self.filter_taps.size % self.num_channels != 0:
            raise ValueError(f"Filter length ({self.filter_taps.size}) must be divisible by num_channels ({self.num_channels})")

        self.OLA= int(self.filter_taps.size // self.num_channels)
        
        self.banks_mat = self.filter_taps.reshape(self.OLA, self.num_channels)
       
        self.past_block_data = None
        self.frames_processed = 0
    


    def compute(self, data: ArrayType) -> ArrayType:
        """
        Perform polyphase channelization with overlap using a filter bank structure.
        """
        data = self.type.asarray(data, dtype=self.type.complex64)

        continued_data, num_output_frames = self.assemble_data_and_frames(data)

        if num_output_frames == 0:
            return self.type.empty((0, self.num_channels), dtype=self.type.complex64)

        weighted_sum = self.apply_polyphase_filter_bank(continued_data, num_output_frames)
        output = self.apply_ifft_and_phase_correction(weighted_sum, num_output_frames)

        self.past_block_data = continued_data[-(self.filter_taps.size - 1):].copy()
        self.frames_processed += num_output_frames

        return output.astype(self.type.complex64)


    def assemble_data_and_frames(self, data: ArrayType) -> tuple[ArrayType, int]:
        """
        Concatenate history with current block and determine how many
        output frames can be produced.
        """

        if self.past_block_data is None:
            continued_data = data
        else:
            continued_data = self.type.concatenate([self.past_block_data, data])

        first_time_index = self.filter_taps.size - 1

        num_output_frames = ((continued_data.size - 1) - first_time_index) // self.decimation_factor + 1

        return continued_data, num_output_frames


    def apply_polyphase_filter_bank(self, continued_data: ArrayType, num_output_frames: int) -> ArrayType:
        """
        This function performs the channelization step by applying an IFFT
        across the polyphase columns and correcting phase offsets caused by
        decimation. 
        """
        weighted_sum = self.type.zeros(
            (num_output_frames, self.num_channels),
            dtype=self.type.complex64
        )

        first_time_index = self.filter_taps.size - 1
        frame_time_idx = first_time_index + self.type.arange(num_output_frames) * self.decimation_factor
        channel_idx = self.type.arange(self.num_channels)

        for p in range(self.OLA):
            sample_idx = frame_time_idx[:, None] - (p * self.num_channels + channel_idx[None, :])
            sub_mat = continued_data[sample_idx]
            weighted_sum += sub_mat * self.banks_mat[p, :]

        return weighted_sum


    def apply_ifft_and_phase_correction(self, weighted_sum: ArrayType, num_output_frames: int) -> ArrayType:
        """
        Apply IFFT, phase correction, and scale to produce final output.
        """
        ifft_out = self.type.fft.ifft(weighted_sum, axis=1)

        n_frame = self.type.arange(num_output_frames).reshape(-1, 1) + self.frames_processed
        k_idx = self.type.arange(self.num_channels).reshape(1, -1)

        phase_corr = self.type.exp(-1j * 2 * np.pi * (self.decimation_factor / self.num_channels) * k_idx * n_frame         )

        output = ifft_out * phase_corr * self.num_channels
        return output