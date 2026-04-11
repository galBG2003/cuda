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
from logics.computes.simple_channelizer import simpleChannelizer


class ChannelizerBenchmarkConfig(pydantic.BaseModel):
    polyphase_config: polyphaseChannelizerOverlap.Config | None = None
    simple_config: simpleChannelizer.Config | None = None
    timing_csv_path: str = "benchmark_results/channelizer_timing_results.csv"
    output_dir: str = "compute_output"
    random_seed: int = 0


@pytest.fixture(scope="session")
def test_config(request) -> ChannelizerBenchmarkConfig:
    scenario_name = request.param
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name=scenario_name)

    cfg_dict = OmegaConf.to_container(cfg.test_config, resolve=True)
    return ChannelizerBenchmarkConfig.model_validate(cfg_dict)


def generate_random_complex_signal(num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(num_samples, dtype=np.float32)
    imag = rng.standard_normal(num_samples, dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


def write_complex_binary(data: np.ndarray, filename: str) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(data, dtype=np.complex64)
    flat = data.ravel(order="F")

    interleaved = np.empty(flat.size * 2, dtype=np.float32)
    interleaved[0::2] = flat.real
    interleaved[1::2] = flat.imag
    interleaved.tofile(path)


def append_timing_result(
    csv_path: str,
    *,
    implementation: str,
    device: str,
    input_signal: np.ndarray,
    calc_output: np.ndarray,
    elapsed_seconds: float,
    num_channels: int,
    decimation_factor: int | None,
    num_samples: int,
) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    input_size_gb = input_signal.nbytes / (1024 ** 3)
    output_size_gb = calc_output.nbytes / (1024 ** 3)

    row = {
        "implementation": implementation,
        "device": device,
        "num_samples": int(num_samples),
        "input_size_gb": input_size_gb,
        "output_size_gb": output_size_gb,
        "elapsed_seconds": elapsed_seconds,
        "num_channels": int(num_channels),
        "decimation_factor": "" if decimation_factor is None else int(decimation_factor),
        "input_num_samples": int(input_signal.size),
        "output_num_frames": int(calc_output.shape[0]),
    }

    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_module(
    test_config: ChannelizerBenchmarkConfig,
    implementation: str,
    device: str,
):
    use_gpu = (device == "gpu")

    if implementation == "polyphase":
        if test_config.polyphase_config is None:
            raise ValueError("polyphase_config is missing in test config")
        
        # עדכון ה-GPU רק עבור המודול שביקשת שתומך בזה
        test_config.polyphase_config.use_gpu = use_gpu
        module = test_config.polyphase_config.create_logical_instance()
        module.initialize()
        return module

    if implementation == "simple":
        # וידוא שמישהו לא ניסה להריץ simple ב-GPU בטעות דרך ה-parametrize
        if use_gpu:
            raise ValueError("simpleChannelizer is CPU only in this benchmark")
            
        if test_config.simple_config is None:
            raise ValueError("simple_config is missing in test config")
        
        # אנחנו לא נוגעים בשדה use_gpu כאן כי הוא לא קיים ב-Config של ה-simple
        module = test_config.simple_config.create_logical_instance()
        module.initialize()
        return module

    raise ValueError(f"Unknown implementation: {implementation}")


class TestPerformance:
    @pytest.mark.parametrize(
        ("implementation", "device"),
        [
            ("polyphase", "cpu"),
            ("polyphase", "gpu"),
            ("simple", "cpu"),
        ],
    )
    @pytest.mark.parametrize(
        "num_samples",
        [
            10_000,
            100_000,
            1_000_000,
            5000000,
            10_000_000,
            30000000,
            70000000
        ],
    )
    @pytest.mark.parametrize(
        "test_config",
        ["test_channelizer_benchmark_config"],
        indirect=True,
    )
    def test_channelizer_performance_full_input(
        self,
        test_config: ChannelizerBenchmarkConfig,
        implementation: str,
        device: str,
        num_samples: int,
    ) -> None:
        # Optional safety limit for simpleChannelizer memory usage
        # if implementation == "simple"  :
        #     pytest.skip("simpleChannelizer is skipped for very large inputs due to memory usage")

        module = build_module(test_config, implementation, device)

        input_signal = generate_random_complex_signal(
            num_samples=num_samples,
            seed=test_config.random_seed,
        )

        if device == "gpu":
            cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()

        calc_output = module.compute(input_signal)

        if device == "gpu":
            cp.cuda.Stream.null.synchronize()
            calc_output = cp.asnumpy(calc_output)

        total_compute_seconds = time.perf_counter() - t0

        assert calc_output.ndim == 2, "Output must be 2D"
        num_channels = calc_output.shape[1]

        decimation_factor = getattr(module, "decimation_factor", None)

        output_dir = Path(test_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{implementation}_{device}_{num_samples}.32fc"
        write_complex_binary(calc_output, str(output_file))

        append_timing_result(
            test_config.timing_csv_path,
            implementation=implementation,
            device=device,
            input_signal=input_signal,
            calc_output=calc_output,
            elapsed_seconds=total_compute_seconds,
            num_channels=num_channels,
            decimation_factor=decimation_factor,
            num_samples=num_samples,
        )

        throughput_gbps = input_signal.nbytes / total_compute_seconds / (1024 ** 3)

        print(
            f"\nImplementation: {implementation} | "
            f"Device: {device} | "
            f"Samples: {num_samples} | "
            f"Input size: {input_signal.nbytes / (1024 ** 3):.6f} GB | "
            f"Compute time: {total_compute_seconds:.6f} s | "
            f"Throughput: {throughput_gbps:.6f} GB/s | "
            f"Output file: {output_file}"
        )