import numpy as np
import pydantic
import pytest
import hydra
from omegaconf import OmegaConf

from utils.signal_generate import generate_test_input
from logics.computes.simple_channelizer import simpleChannelizer


class ChannelizerTestConfig(pydantic.BaseModel):
    channelizer_config: simpleChannelizer.Config
    tol: pydantic.PositiveFloat = 1e-2
  
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
        module = test_config.channelizer_config.create_logical_instance()
        fs = test_config.channelizer_config.fs_hz
        bw = test_config.channelizer_config.bw_hz
        num_taps = test_config.channelizer_config.num_taps
        num_channels = int(fs // bw)
        
        # Increase duration to give the filter time to settle
        channel_duration = 1.0 
        num_samples = int(fs * channel_duration * num_channels)

        input_signal, expected_amplitudes = generate_test_input(fs, bw, num_channels, num_samples)
        calc_output = module.compute(input_signal)

        # The delay of an FIR filter is half the number of taps
        filter_delay_samples = (num_taps - 1) // 2
        # Adjusted delay for the decimated output
        decimated_delay = filter_delay_samples // num_channels

        samples_per_channel_decimated = (num_samples // num_channels) // num_channels
        
        errors = []
        
        for i in range(num_channels):
            # Define the window where this channel was actually broadcasting
            # We add the decimated_delay to account for the filter's latency
            start = i * samples_per_channel_decimated + decimated_delay
            end = (i + 1) * samples_per_channel_decimated + decimated_delay
            
            # To get a clean reading, ignore the edges (first and last 20% of the burst)
            # This avoids the "ramp up" and "ramp down" of the filter
            margin = int(0.2 * samples_per_channel_decimated)
            active_region = calc_output[i, start + margin : end - margin]
            
            if active_region.size == 0:
                errors.append(f"Channel {i}: Active region is empty. Check durations.")
                continue

            # Measure peak or mean of the steady-state portion
            measured_amplitude = np.mean(np.abs(active_region))
            expected = expected_amplitudes[i]
            
            # Use a relative tolerance
            relative_error = abs(measured_amplitude - expected) / expected

            if relative_error > test_config.tol:
                errors.append(
                    f"Channel {i}: Expected {expected:.3f}, got {measured_amplitude:.3f} "
                    f"(Error: {relative_error:.2%})"
                )

        assert not errors, "\n".join(errors)