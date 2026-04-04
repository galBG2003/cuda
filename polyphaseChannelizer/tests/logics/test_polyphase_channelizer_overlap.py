import numpy as np
import pydantic
import pytest
import hydra
import cupy as cp
from omegaconf import OmegaConf

from logics.computes.polyphase_channlezier_overlap import PolyphaseChannelizerOverlap
from utils.cosine_similarity import cosine_similarity

class ChannelizerTestConfig(pydantic.BaseModel):
    channelizer_config: PolyphaseChannelizerOverlap.Config
    test_signal_path: str
    input_signal_path: str
    tol: pydantic.PositiveFloat = 0.05
    block_size: pydantic.PositiveInt = 5000

@pytest.fixture(scope="session")
def test_config(request) -> ChannelizerTestConfig:
    scenario_name = request.param
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name=scenario_name)
    test_cfg_dict = OmegaConf.to_container(cfg.test_config, resolve=True)
    test_cfg = ChannelizerTestConfig.model_validate(test_cfg_dict)
    return test_cfg

class TestClass:
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize("test_config", ["test_polyphase_channelizer_overlap_config"], indirect=True)
    def test_polyphase_channelizer_overlap(self, test_config: ChannelizerTestConfig, device: str):
        
        test_config.channelizer_config.use_gpu = (device == "gpu")
        module = test_config.channelizer_config.create_logical_instance()
        module.initialize()

        num_channels = module.num_channels
        decimation_factor = module.decimation_factor
        block_size = test_config.block_size

        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        test_signal_flat = np.fromfile(test_config.test_signal_path, dtype=np.complex64)
        
        test_signal_channels = test_signal_flat.reshape(-1, num_channels, order="F")

        block_outputs = []
        for start in range(0, input_signal.size, block_size):
            end = min(start + block_size, input_signal.size)
            block = input_signal[start:end]

            block_out = module.compute(block)

            if device == "gpu":
                block_out = cp.asnumpy(block_out)

            if block_out.shape[0] > 0:
                block_outputs.append(block_out)

        if len(block_outputs) == 0:
            pytest.fail("Blocked processing produced no output frames.")

        calc_output = np.vstack(block_outputs)
        
        data_to_save = calc_output.T.copy().astype(np.complex64)
        data_to_save.tofile("calc_output30.bin")

        assert calc_output.ndim == 2, "Final concatenated output must be 2D"
        assert calc_output.shape[1] == num_channels, (
            f"Final channel count mismatch. Expected {num_channels}, got {calc_output.shape[1]}"
        )

        margin = int(np.ceil(module.filter_taps.size / decimation_factor))

        min_frames = min(test_signal_channels.shape[0], calc_output.shape[0])
        assert min_frames > 2 * margin, (
            f"Not enough frames after blocking to trim margins. "
            f"min_frames={min_frames}, margin={margin}"
        )

        test_trimmed = test_signal_channels[margin:min_frames - margin, :]
        calc_trimmed = calc_output[margin:min_frames - margin, :]
    
        similarity = cosine_similarity(test_trimmed, calc_trimmed)

        active_channels = np.arange(0, num_channels, 2)
        relevant_similarity = similarity[active_channels]
        
        print(f"\nSimilarities for active channels: {relevant_similarity}")
        
        assert np.all(relevant_similarity > 1 - test_config.tol), (
            f"Failed on active channels after blocked processing.\n"
            f"Indices: {active_channels}\n"
            f"Similarities: {relevant_similarity}"
        )