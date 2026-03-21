import numpy as np
import pydantic
import pytest
import hydra
from omegaconf import OmegaConf

from logics.computes.polyphase_Channelizer import polyphaseChannelizer
from utils.signal_generator import generate_test_input
from utils.cosine_similarity import cosine_similarity
from scipy.io import loadmat


class TestConfig(pydantic.BaseModel):
    channelizer_config: polyphaseChannelizer.Config
    test_signal_path: str
    input_signal_path:str
    tol: pydantic.PositiveFloat = 0.05

@pytest.fixture(scope="session")
def test_config(request) -> TestConfig:
    scenario_name = request.param
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name=scenario_name)
    test_cfg_dict = OmegaConf.to_container(cfg.test_config, resolve=True)
    test_cfg = TestConfig.model_validate(test_cfg_dict)
    return test_cfg

def generate_single_tone(fs, freq_hz, num_samples, amp=1.0):
    t = np.arange(num_samples) / fs
    return (amp * np.exp(1j * 2 * np.pi * freq_hz * t)).astype(np.complex64)


class TestClass:
    @pytest.mark.parametrize("test_config", ["test_polyphase_channelizer_config"], indirect=True)
    def test_polyphase_channelizer(self, test_config: TestConfig):
        fs = test_config.channelizer_config.fs_hz
        bw = test_config.channelizer_config.bw_hz
        num_channels = int(fs // bw)

        input_signal = np.fromfile(test_config.input_signal_path, dtype=np.complex64)
        test_signal_flat = np.fromfile(test_config.test_signal_path, dtype=np.complex64)
        num_frames = len(test_signal_flat) // num_channels
        test_signal_channels = np.zeros((num_frames, num_channels), dtype=np.complex64)
        for i in range(num_channels):
            start = i * num_frames
            end = (i + 1) * num_frames
            test_signal_channels[:, i] = test_signal_flat[start:end]
        
        module = test_config.channelizer_config.create_logical_instance()
        calc_output = module.compute(input_signal)
        assert calc_output.ndim == 2, "Output must be 2D"
        assert calc_output.shape[1] == num_channels, "Channel count mismatch"

        similarity = cosine_similarity(test_signal_channels, calc_output)
        assert np.all(similarity > 1 - test_config.tol), (
            f"Similarity per channel: {similarity}\n"
            f"Failed channels: {np.where(similarity <= 1 - test_config.tol)[0]}"
        )
            