from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from time import perf_counter_ns
from typing import Any

import torch
from pydantic import BaseModel

from metalbench import env
from metalbench.correctness import tree_to_device
from metalbench.import_utils import (
    import_module_from_path,
    require_generated_contract,
    require_reference_contract,
)


class TimingResult(BaseModel):
    ok: bool
    device: str
    median_ms: float | None
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None
    p25_ms: float | None
    p75_ms: float | None
    warmup: int
    trials: int
    errors: list[str]


def time_generated_metal_kernel(
    ref_path: Path,
    kernel_path: Path,
    warmup: int = 10,
    trials: int = 100,
    require_mps: bool = True,
) -> TimingResult:
    if not env.is_mps_available():
        return _handle_missing_mps(warmup=warmup, require_mps=require_mps)

    reference_module = import_module_from_path(ref_path, "metalbench_reference")
    generated_module = import_module_from_path(kernel_path, "metalbench_generated")
    require_reference_contract(reference_module)
    require_generated_contract(generated_module)

    mps_device = torch.device("mps")
    try:
        init_inputs = _as_args(reference_module.get_init_inputs())
        inputs = _as_args(reference_module.get_inputs())
        model = generated_module.ModelNew(*tree_to_device(init_inputs, mps_device))
        model.to(mps_device).eval()
        mps_inputs = tree_to_device(inputs, mps_device)
        samples = _time_forward_passes(
            lambda: model(*mps_inputs),
            warmup=warmup,
            trials=trials,
        )
    except Exception as exc:  # noqa: BLE001
        return _failure_result(warmup=warmup, errors=[str(exc)])

    return _summarize_timings_ms(samples, warmup=warmup, errors=[])


def time_reference_mps_baseline(
    ref_path: Path,
    warmup: int = 10,
    trials: int = 100,
    require_mps: bool = True,
) -> TimingResult:
    if not env.is_mps_available():
        return _handle_missing_mps(warmup=warmup, require_mps=require_mps)

    reference_module = import_module_from_path(ref_path, "metalbench_reference")
    require_reference_contract(reference_module)

    mps_device = torch.device("mps")
    try:
        init_inputs = _as_args(reference_module.get_init_inputs())
        inputs = _as_args(reference_module.get_inputs())
        model = reference_module.Model(*init_inputs).to(mps_device)
        model.eval()
        mps_inputs = tree_to_device(inputs, mps_device)
        samples = _time_forward_passes(
            lambda: model(*mps_inputs),
            warmup=warmup,
            trials=trials,
        )
    except Exception as exc:  # noqa: BLE001
        return _failure_result(warmup=warmup, errors=[str(exc)])

    return _summarize_timings_ms(samples, warmup=warmup, errors=[])


def _time_forward_passes(
    forward: Callable[[], Any],
    *,
    warmup: int,
    trials: int,
) -> list[float]:
    samples: list[float] = []

    with torch.no_grad():
        for _ in range(warmup):
            forward()
            torch.mps.synchronize()

        for _ in range(trials):
            torch.mps.synchronize()
            start_ns = perf_counter_ns()
            forward()
            torch.mps.synchronize()
            elapsed_ns = perf_counter_ns() - start_ns
            samples.append(elapsed_ns / 1_000_000)

    return samples


def _summarize_timings_ms(
    samples_ms: list[float],
    *,
    warmup: int,
    errors: list[str],
) -> TimingResult:
    if not samples_ms:
        return _failure_result(warmup=warmup, errors=errors)

    sorted_samples = sorted(samples_ms)
    return TimingResult(
        ok=not errors,
        device="mps",
        median_ms=median(sorted_samples),
        mean_ms=mean(sorted_samples),
        min_ms=min(sorted_samples),
        max_ms=max(sorted_samples),
        p25_ms=_percentile(sorted_samples, 0.25),
        p75_ms=_percentile(sorted_samples, 0.75),
        warmup=warmup,
        trials=len(samples_ms),
        errors=errors,
    )


def _percentile(sorted_samples: list[float], quantile: float) -> float:
    position = (len(sorted_samples) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_samples) - 1)
    weight = position - lower_index
    lower = sorted_samples[lower_index]
    upper = sorted_samples[upper_index]
    return lower + (upper - lower) * weight


def _handle_missing_mps(*, warmup: int, require_mps: bool) -> TimingResult:
    if require_mps:
        env.require_mps()
    return _failure_result(
        warmup=warmup,
        errors=["MPS is not available; timing trials were skipped."],
    )


def _failure_result(*, warmup: int, errors: list[str]) -> TimingResult:
    return TimingResult(
        ok=False,
        device="mps",
        median_ms=None,
        mean_ms=None,
        min_ms=None,
        max_ms=None,
        p25_ms=None,
        p75_ms=None,
        warmup=warmup,
        trials=0,
        errors=errors,
    )


def _as_args(obj: Any) -> tuple[Any, ...]:
    if isinstance(obj, tuple):
        return obj
    if isinstance(obj, list):
        return tuple(obj)
    return (obj,)


__all__ = [
    "TimingResult",
    "time_generated_metal_kernel",
    "time_reference_mps_baseline",
]
