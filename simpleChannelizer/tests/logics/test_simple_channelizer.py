import numpy as np
import pydantic
import pytest
import hydra
from omegaconf import OmegaConf

from utils.signal_generate import generate_test_input
from logics.computes.simple_channelizer import simpleChannelizer
from utils.cosine_similarity import cosine_similarity

class ChannelizerTestConfig(pydantic.BaseModel):
    channelizer_config: simpleChannelizer.Config
    tol: pydantic.PositiveFloat = 1e-2
    input_path: str
    test_path: str
  
@pytest.fixture(scope="session")
def test_config(request) -> ChannelizerTestConfig:
    scenario_name = request.param
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name=scenario_name)
    test_cfg_dict = OmegaConf.to_container(cfg.test_config, resolve=True)
    test_cfg = ChannelizerTestConfig.model_validate(test_cfg_dict)
    return test_cfg

class TestClass:
    @pytest.mark.parametrize("test_config", ["test_simple_channelizer_config"], indirect=True)
    def test_extract_channel(self, test_config: ChannelizerTestConfig):
        fs = test_config.channelizer_config.fs_hz
        bw = test_config.channelizer_config.bw_hz
        num_channels = int(fs // bw)

        input_signal = np.fromfile(test_config.input_path, dtype=np.complex64)
        test_signal_flat = np.fromfile(test_config.test_path, dtype=np.complex64)
        num_frames = len(test_signal_flat) // num_channels
        test_signal_channels = np.zeros((num_channels,num_frames), dtype=np.complex64)
        for i in range(num_channels):
            start = i * num_frames
            end = (i + 1) * num_frames
            test_signal_channels[i, :] = test_signal_flat[start:end]
        
        module = test_config.channelizer_config.create_logical_instance()
        calc_output = module.compute(input_signal)
        assert calc_output.ndim == 2, "Output must be 2D"
        assert calc_output.shape[0] == num_channels, "Channel count mismatch"

        similarity = cosine_similarity(test_signal_channels, calc_output)
        assert np.all(similarity > 1 - test_config.tol), (
            f"Similarity per channel: {similarity}\n"
            f"Failed channels: {np.where(similarity <= 1 - test_config.tol)[0]}"
        )