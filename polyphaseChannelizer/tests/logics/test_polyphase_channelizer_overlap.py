import csv
import time
from pathlib import Path

import numpy as np
import pydantic
import pytest
import hydra
import cupy as cp
from omegaconf import OmegaConf

from logics.computes.polyphase_channlezier_overlap import polyphaseChannelizerOverlap
from utils.cosine_similarity import cosine_similarity
from utils.write_complex_to_binary_file import write_complex_binary


class ChannelizerTestConfig(pydantic.BaseModel):
    channelizer_config: polyphaseChannelizerOverlap.Config
    test_signal_path: str
    input_signal_path: str
    tol: pydantic.PositiveFloat = 0.05
    block_size: pydantic.PositiveInt = 5000
    timing_csv_path: str = "benchmark_results/channelizer_timing_results.csv"
    output_file_path: str = "channelizer_output.32fc"


@pytest.fixture(scope="session")
def test_config(request) -> ChannelizerTestConfig:
    scenario_name = request.param
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name=scenario_name)

    test_cfg_dict = OmegaConf.to_container(cfg.test_config, resolve=True)
    test_cfg = ChannelizerTestConfig.model_validate(test_cfg_dict)
    return test_cfg


def append_timing_result(
    csv_path: str,
    *,
    device: str,
    input_signal: np.ndarray,
    calc_output: np.ndarray,
    elapsed_seconds: float,
    block_size: int,
    num_channels: int,
    decimation_factor: int,
) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    input_size_gb = input_signal.nbytes / (1024 ** 3)
    output_size_gb = calc_output.nbytes / (1024 ** 3)

    row = {
        "device": device,
        "input_size_gb": input_size_gb,
        "output_size_gb": output_size_gb,
        "elapsed_seconds": elapsed_seconds,
        "block_size": block_size,
        "num_channels": num_channels,
        "decimation_factor": decimation_factor,
        "input_num_samples": int(input_signal.size),
        "output_num_frames": int(calc_output.shape[0]),
    }

    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class TestClass:
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    @pytest.mark.parametrize(
        "test_config",
        ["test_polyphase_channelizer_overlap_config"],
        indirect=True,
    )
    def test_polyphase_channelizer_overlap(
        self,
        test_config: ChannelizerTestConfig,
        device: str,
    ) -> None:
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

        if device == "gpu":
            cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()

        for start in range(0, input_signal.size, block_size):
            end = min(start + block_size, input_signal.size)
            block = input_signal[start:end]

            block_out = module.compute(block)

            if device == "gpu":
                cp.cuda.Stream.null.synchronize()
                block_out = cp.asnumpy(block_out)

            assert block_out.ndim == 2, "Each block output must be 2D"
            assert block_out.shape[1] == num_channels, (
                f"Channel count mismatch in block. Expected {num_channels}, got {block_out.shape[1]}"
            )

            if block_out.shape[0] > 0:
                block_outputs.append(block_out)

        if device == "gpu":
            cp.cuda.Stream.null.synchronize()

        total_compute_seconds = time.perf_counter() - t0

        if len(block_outputs) == 0:
            pytest.fail("Blocked processing produced no output frames.")

        calc_output = np.vstack(block_outputs)

        assert calc_output.ndim == 2, "Final concatenated output must be 2D"
        assert calc_output.shape[1] == num_channels, (
            f"Final channel count mismatch. Expected {num_channels}, got {calc_output.shape[1]}"
        )

        from pathlib import Path

        output_dir = Path("compute_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"channelizer_{device}_{input_signal.size}.32fc"
        write_complex_binary(calc_output, str(output_file))

        append_timing_result(
            "benchmark_results/test.csv",
            device=device,
            input_signal=input_signal,
            calc_output=calc_output,
            elapsed_seconds=total_compute_seconds,
            block_size=block_size,
            num_channels=num_channels,
            decimation_factor=decimation_factor,
        )

        print(
            f"\nDevice: {device} | "
            f"Input size: {input_signal.nbytes / (1024 ** 3):.6f} GB | "
            f"Total blocked compute time: {total_compute_seconds:.6f} s"
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

        assert np.all(relevant_similarity > 1 - test_config.tol), (
            f"Failed on active channels after blocked processing.\n"
            f"Indices: {active_channels}\n"
            f"Similarities: {relevant_similarity}"
        )