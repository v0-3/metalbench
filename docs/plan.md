# Codex-Ready Plan: `metalbench` with `uv`

`metalbench` is a new Python project for **Apple Metal-only kernel generation and testing**. It uses KernelBench reference problems as PyTorch task specifications, but it does **not** use KernelBench CUDA, Triton, HIP, ROCm, Modal, NVIDIA, or AMD backend paths.

KernelBench supplies problem definitions and metric inspiration. `metalbench` supplies the Metal/MPS adapter, static validation, correctness testing, MPS timing, and Metal-only analysis.

This version standardizes the project workflow on `uv`: dependency changes use `uv add` / `uv remove`, the lockfile is `uv.lock`, environment setup uses `uv sync`, and project commands run through `uv run`. It also standardizes KernelBench reference loading on a project-root `KernelBench/` problem directory, for example `./KernelBench/level1/19_ReLU.py`.

---

## 1. Project objective

Create a Python project named `metalbench` that can:

- Load KernelBench-style reference PyTorch problems from the project-root `KernelBench/` directory.
- Accept generated `ModelNew` Python files that use Apple Metal only.
- Statically validate generated kernels for Metal-only constraints.
- Check correctness against the PyTorch reference model.
- Time generated kernels on `torch.device("mps")`.
- Time the PyTorch reference model on `mps` when supported.
- Report Metal-only metrics:
  - `metal_fast_0`
  - `metal_fast_1`
  - `metal_fast_2`
  - speedup versus PyTorch MPS baseline

---

## 2. Hard scope constraints

### Included

- Apple Silicon
- macOS
- PyTorch MPS
- Metal Shading Language
- `torch.mps.compile_shader(...)`
- `torch.mps.load_metallib(...)`
- CPU PyTorch correctness oracle
- PyTorch MPS performance baseline
- KernelBench task/spec loading

### Excluded

- CUDA
- Triton
- HIP / ROCm
- CUTE
- TileLang
- ThunderKittens
- Modal GPU evaluation
- NVIDIA benchmarking
- AMD benchmarking
- CPU fallback as a performance path

---

## 3. Codex implementation prompt

Paste this into Codex as the primary instruction.

```text
Create a new Python project named `metalbench`.

The project is a Metal-only evaluation harness for generated Apple Metal kernels, using KernelBench problems from the project-root `KernelBench/` directory as reference PyTorch specs. Do not implement or depend on CUDA, Triton, HIP, ROCm, CUTE, TileLang, ThunderKittens, Modal, or NVIDIA/AMD-specific paths.

Primary goal:
- Load KernelBench reference problems from the project-root `KernelBench/` directory.
- Generate or accept KernelBench-compatible `ModelNew` Python files that use Apple Metal only.
- Statically validate generated kernels for Metal-only constraints.
- Check correctness against the PyTorch reference model.
- Time generated kernels on `torch.device("mps")`.
- Time the PyTorch reference model on `mps` when supported.
- Report Metal-only metrics: `metal_fast_0`, `metal_fast_1`, `metal_fast_2`, and speedup versus PyTorch MPS baseline.

Hard constraints:
- Backend is always Apple Metal.
- Runtime device is always `mps` for generated kernels.
- Correctness oracle may use CPU PyTorch reference outputs.
- Performance baseline must be PyTorch on `mps`, never CPU.
- Generated kernels must use `torch.mps.compile_shader(...)` or `torch.mps.load_metallib(...)`.
- Generated kernels must define `class ModelNew(torch.nn.Module)`.
- Generated kernels must not use `torch.cuda`, Triton, HIP, ROCm, CuPy, PyCUDA, nvcc, CUDA C++, or CPU fallback as an implementation path.
- Generated kernels must not call the reference `Model`.
- Generated kernels must not use high-level PyTorch ops as the primary computation path, except for safe allocation/shape/device utilities such as `torch.empty_like`, `torch.empty`, `.contiguous()`, `.to("mps")`, and tensor metadata reads.

Important runtime caveat:
- Some Codex environments may not have Apple Silicon or MPS.
- Implement non-MPS unit tests that run anywhere.
- Implement MPS integration tests that are skipped unless `torch.backends.mps.is_available()` is true.
- Do not fake MPS benchmark results.
- Add a CLI flag `--require-mps`; when set, fail immediately if MPS is unavailable.

Project and dependency management:
- Use `uv` as the only project, dependency, lockfile, virtual environment, and command runner workflow.
- Use `pyproject.toml` plus `uv.lock`; commit both files.
- Prefer standard Python packaging with `src/metalbench`.
- Support Python 3.10+ and pin the local interpreter with `.python-version` or `uv python pin 3.10`.
- Runtime dependencies: `torch`, `numpy`, `pandas`, `pydantic>=2`, `typer`, `rich`.
- Development dependencies belong in the `dev` dependency group: `pytest`, `ruff`, `mypy`.
- Optional dependency group / extra: `hf` with `datasets`, only if adding Hugging Face dataset support.
- Use `uv add ...`, `uv add --dev ...`, and `uv add --optional hf ...` to mutate dependencies.
- Use `uv sync`, `uv run ...`, and `uv run --locked ...` instead of `python -m pip install ...` or manually activated virtual environments.
```

---

## 4. Required repository structure

```text
metalbench/
  pyproject.toml
  uv.lock
  .python-version
  README.md
  LICENSE
  .gitignore
  KernelBench/              # project-root KernelBench reference-problem directory
    level1/
    level2/
    level3/
    level4/
  .agents/
    skills/
      metal-kernel-generation/
        SKILL.md
  src/
    metalbench/
      __init__.py
      __main__.py
      cli.py
      config.py
      types.py
      env.py
      import_utils.py
      kernelbench_adapter.py
      static_check.py
      correctness.py
      timing.py
      eval_one.py
      eval_batch.py
      analysis.py
      generation_contract.py
      templates/
        __init__.py
        relu_compile_shader.py
        square_compile_shader.py
  tests/
    test_env.py
    test_static_check.py
    test_import_utils.py
    test_analysis.py
    test_cli_smoke.py
    test_mps_runtime.py
  examples/
    generated/
      level_1_problem_synthetic_relu_sample_0_kernel.py
    references/
      synthetic_relu_reference.py
  runs/
    .gitkeep
```

Add a console entry point:

```toml
[project.scripts]
metalbench = "metalbench.cli:app"
```

---

## 5. Phase 1: `uv` project scaffolding

Initialize the project with `uv` and keep `uv.lock` under version control.

Recommended creation commands:

```bash
uv init --package metalbench --python 3.10
cd metalbench
uv python pin 3.10
uv add torch numpy pandas "pydantic>=2" typer rich
uv add --dev pytest ruff mypy
uv add --optional hf datasets
uv lock
```

If Codex creates the files directly instead of running commands, create an equivalent `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "metalbench"
version = "0.1.0"
description = "Apple Metal-only kernel generation and evaluation harness for KernelBench-style tasks"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "torch",
  "numpy",
  "pandas",
  "pydantic>=2",
  "typer",
  "rich",
]

[project.optional-dependencies]
hf = ["datasets"]

[dependency-groups]
dev = [
  "pytest",
  "ruff",
  "mypy",
]

[project.scripts]
metalbench = "metalbench.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/metalbench"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "mps: tests requiring Apple Metal MPS runtime",
]
```

Create `.python-version`:

```text
3.10
```

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
*.pyc
.ruff_cache/
.mypy_cache/
.pytest_cache/
runs/*
!runs/.gitkeep
*.metallib
.DS_Store
```

`uv` workflow rules for this repo:

```text
- Do not use `python -m pip install -e ...` in project docs or acceptance gates.
- Do not require manual `source .venv/bin/activate` for normal commands.
- Use `uv sync` to create/sync the project environment.
- Use `uv run <command>` for CLI, tests, linting, and type checks.
- Use `uv run --locked <command>` in CI/reproducibility checks.
- Commit `uv.lock` after dependency changes.
```

Populate the project-root `KernelBench/` problem directory before testing real KernelBench tasks. In this plan, `./KernelBench` is the benchmark dataset-file directory and must contain `level1/`, `level2/`, `level3/`, and `level4/` directly.

If you have a full upstream KernelBench repository checkout elsewhere, copy or symlink only its benchmark dataset directory into the `metalbench` project root:

```bash
# From the metalbench project root:
mkdir -p KernelBench
rsync -a /path/to/upstream/KernelBench/KernelBench/ ./KernelBench/

# Expected:
ls KernelBench/level1
```

Do not use `third_party/KernelBench` as the default reference location. Do not clone the full upstream repository directly into `./KernelBench` unless the adapter is explicitly changed to handle the nested `KernelBench/KernelBench/level*` layout.

Acceptance criteria:

```bash
uv sync --all-extras
uv run pytest tests/test_env.py tests/test_static_check.py
uv run metalbench --help
uv run ruff check .
uv run mypy src
```


## 6. Phase 2: environment detection

Implement `src/metalbench/env.py`.

Required functions:

```python
def is_mps_available() -> bool:
    ...

def require_mps() -> None:
    ...

def describe_environment() -> dict:
    ...
```

`describe_environment()` must return:

```json
{
  "python": "...",
  "platform": "...",
  "torch_version": "...",
  "mps_available": true,
  "mps_built": true,
  "device": "mps"
}
```

CLI commands:

```bash
uv run metalbench env
uv run metalbench env --require-mps
```

Behavior:

- `metalbench env` prints environment JSON.
- `metalbench env --require-mps` exits nonzero if MPS is unavailable.
- Do not mention or inspect CUDA.

Tests:

- `test_env.py` validates shape of returned dict.
- Tests must not require actual MPS.

---

## 7. Phase 3: project-root KernelBench adapter

Implement `src/metalbench/kernelbench_adapter.py`.

Purpose:

- Load KernelBench reference problems from the project-root `KernelBench/` problem directory.
- Do not import CUDA backend code.
- Prefer project-local file loading over remote dataset access.
- Support Level 1 and Level 2 first.
- Leave Level 3/4 support structurally possible but not required for v0.1.

Data model in `types.py`:

```python
from pydantic import BaseModel
from pathlib import Path

class KBProblem(BaseModel):
    level: int
    problem_id: int
    name: str
    path: Path
    source: str
    source_sha256: str
```

Required functions:

```python
def find_problem_file(level: int, problem_id: int, kernelbench_dir: Path = Path("KernelBench")) -> Path:
    """
    Find ./KernelBench/level{level}/{problem_id}_*.py when kernelbench_dir is the default root-level problem directory.
    """


def load_problem(level: int, problem_id: int, kernelbench_dir: Path = Path("KernelBench")) -> KBProblem:
    ...


def list_problems(level: int, kernelbench_dir: Path = Path("KernelBench")) -> list[KBProblem]:
    ...
```

Expected local KernelBench problem layout:

```text
<metalbench project root>/
  KernelBench/
    level1/
      1_*.py
      2_*.py
      ...
    level2/
    level3/
    level4/
```

`KernelBench/` here is the benchmark dataset-file directory copied or symlinked into the `metalbench` project root. It is not the full upstream repository root and should not require a nested `KernelBench/KernelBench/` path.

Use glob patterns:

```python
kernelbench_dir / f"level{level}" / f"{problem_id}_*.py"
```

Default `kernelbench_dir` should be `Path("KernelBench")`.


CLI commands:

```bash
uv run metalbench kb list --kernelbench-dir KernelBench --level 1
uv run metalbench kb show --kernelbench-dir KernelBench --level 1 --problem-id 19
```

Acceptance criteria:

- If the project-root `KernelBench/` directory is absent, command fails with a clear message.
- If multiple files match, fail clearly.
- `show` prints metadata and path, not the full code by default.
- Add `--print-source` to print the reference source.

---

## 8. Phase 4: dynamic module import utilities

Implement `src/metalbench/import_utils.py`.

Required functions:

```python
def import_module_from_path(path: Path, module_name: str):
    ...


def require_reference_contract(module) -> None:
    """
    Reference module must expose:
    - class Model
    - function get_inputs()
    - function get_init_inputs()
    """


def require_generated_contract(module) -> None:
    """
    Generated module must expose:
    - class ModelNew
    """
```

Tests:

- Create tiny synthetic modules in temporary files.
- Verify import success.
- Verify contract errors are explicit.

---

## 9. Phase 5: Metal static checker

Implement `src/metalbench/static_check.py`.

The static checker must inspect generated Python files before runtime.

Data model:

```python
class StaticCheckResult(BaseModel):
    path: Path
    ok: bool
    errors: list[str]
    warnings: list[str]
    uses_compile_shader: bool
    uses_load_metallib: bool
    found_metal_kernel_source: bool
```

Required function:

```python
def check_generated_metal_kernel(path: Path) -> StaticCheckResult:
    ...
```

Reject these patterns:

```text
torch.cuda
cuda
CUDA
triton
tl.
hip
rocm
ROCm
cupy
pycuda
nvcc
__global__
__device__
__host__
.cu
```

Reject suspicious reference/fallback patterns:

```text
class Model(
Model(
ref_model
reference_model
torch.nn.functional.relu
torch.relu
torch.matmul
torch.mm
torch.bmm
torch.conv
torch.nn.functional.conv
```

Required tokens:

```text
class ModelNew
torch.mps.compile_shader OR torch.mps.load_metallib
```

Additional checks:

- If `compile_shader` is used, require visible Metal source containing `kernel void`.
- Warn if `.cpu()` appears.
- Warn if `.numpy()` appears.
- Warn if `torch.empty_like`, `torch.empty`, or `.contiguous()` are absent; these are not always required, but often indicate missing output allocation.
- Warn if `torch.mps.synchronize()` appears inside `forward`; synchronization should be used in timing harness, not in the kernel implementation, unless there is a documented reason.

CLI:

```bash
uv run metalbench check runs/metal_v0/level_1_problem_19_sample_0_kernel.py
```

Exit code:

- `0` if `ok=true`.
- `1` if `ok=false`.

Tests:

- Accept a minimal valid `compile_shader` sample.
- Accept a minimal valid `load_metallib` sample.
- Reject CUDA.
- Reject Triton.
- Reject missing `ModelNew`.
- Reject missing Metal entry point.
- Warn on `.cpu()`.

---

## 10. Phase 6: correctness harness

Implement `src/metalbench/correctness.py`.

Correctness strategy:

- Reference model runs on CPU.
- Generated `ModelNew` runs on MPS.
- Inputs are generated by the KernelBench reference module using `get_inputs()`.
- Init inputs are generated using `get_init_inputs()`.
- Tensor inputs are copied to MPS for generated model.
- Generated outputs are copied back to CPU before comparison.
- Use `torch.testing.assert_close`.
- Support tensor trees: tensor, tuple, list, dict.

Required helpers:

```python
def tree_to_device(obj, device: torch.device):
    ...


def tree_to_cpu(obj):
    ...


def assert_close_tree(actual, expected, *, rtol: float, atol: float) -> None:
    ...


def run_correctness_trials(
    ref_path: Path,
    kernel_path: Path,
    trials: int = 5,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    require_mps: bool = True,
) -> CorrectnessResult:
    ...
```

Data model:

```python
class CorrectnessResult(BaseModel):
    ok: bool
    trials: int
    passed: int
    failed: int
    errors: list[str]
```

Runtime behavior:

- If MPS unavailable and `require_mps=True`, fail immediately.
- If MPS unavailable and `require_mps=False`, mark skipped with a clear message.
- Do not run generated kernels on CPU.

Tests:

- Unit-test `tree_to_device`, `tree_to_cpu`, and `assert_close_tree`.
- Add MPS integration test using synthetic ReLU reference and generated Metal ReLU; skip if MPS unavailable.

---

## 11. Phase 7: timing harness

Implement `src/metalbench/timing.py`.

Timing rules:

- Generated kernels timed on MPS only.
- Reference baseline timed on MPS only.
- Use warmup trials.
- Use `torch.mps.synchronize()` before timing and after each measured forward pass.
- Report median, mean, min, max, p25, p75.
- Do not use CUDA events.
- Do not use CPU baseline for speedup.

Data model:

```python
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
```

Functions:

```python
def time_generated_metal_kernel(
    ref_path: Path,
    kernel_path: Path,
    warmup: int = 10,
    trials: int = 100,
    require_mps: bool = True,
) -> TimingResult:
    ...


def time_reference_mps_baseline(
    ref_path: Path,
    warmup: int = 10,
    trials: int = 100,
    require_mps: bool = True,
) -> TimingResult:
    ...
```

Important:

- `time_reference_mps_baseline` must run `Model(...).to("mps")`.
- If the reference model uses unsupported MPS ops and fails, return `ok=false` with error details.
- A task with no valid MPS reference timing may still have correctness results, but it must not produce speedup metrics.

Tests:

- Non-MPS tests verify percentile calculation and failure behavior.
- MPS test times synthetic ReLU if MPS is available.

---

## 12. Phase 8: evaluate one kernel

Implement `src/metalbench/eval_one.py`.

Data model:

```python
class EvalOneResult(BaseModel):
    run_name: str | None
    level: int | None
    problem_id: int | None
    sample_id: int | None
    ref_path: Path
    kernel_path: Path
    static_check: StaticCheckResult
    correctness: CorrectnessResult | None
    generated_timing: TimingResult | None
    mps_baseline_timing: TimingResult | None
    speedup_vs_mps: float | None
    metal_fast_0: bool
    metal_fast_1: bool
    metal_fast_2: bool
    errors: list[str]
```

Function:

```python
def evaluate_one(
    ref_path: Path,
    kernel_path: Path,
    *,
    run_name: str | None = None,
    level: int | None = None,
    problem_id: int | None = None,
    sample_id: int | None = None,
    correctness_trials: int = 5,
    perf_trials: int = 100,
    warmup: int = 10,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    require_mps: bool = True,
) -> EvalOneResult:
    ...
```

Evaluation order:

```text
1. Static check.
2. Correctness.
3. Generated Metal timing.
4. PyTorch MPS reference timing.
5. Compute speedup and metal_fast metrics.
```

Metric definitions:

```text
metal_fast_0 = static_check.ok and correctness.ok
metal_fast_1 = metal_fast_0 and speedup_vs_mps > 1.0
metal_fast_2 = metal_fast_0 and speedup_vs_mps > 2.0
speedup_vs_mps = mps_baseline_median_ms / generated_metal_median_ms
```

CLI:

```bash
uv run metalbench eval-one \
  --ref-path examples/references/synthetic_relu_reference.py \
  --kernel-path examples/generated/level_1_problem_synthetic_relu_sample_0_kernel.py \
  --correctness-trials 5 \
  --perf-trials 100 \
  --warmup 10 \
  --require-mps
```

Output:

- JSON to stdout by default.
- Optional `--output path.json`.

---

## 13. Phase 9: batch evaluation

Implement `src/metalbench/eval_batch.py`.

Expected generated run layout:

```text
runs/<run_name>/
  level_1_problem_19_sample_0_kernel.py
  level_1_problem_20_sample_0_kernel.py
  eval_results.json
```

Filename parser must support:

```text
level_<level>_problem_<problem_id>_sample_<sample_id>_kernel.py
```

Function:

```python
def evaluate_run_directory(
    run_dir: Path,
    kernelbench_dir: Path = Path("KernelBench"),
    *,
    correctness_trials: int = 5,
    perf_trials: int = 100,
    warmup: int = 10,
    require_mps: bool = True,
) -> list[EvalOneResult]:
    ...
```

CLI:

```bash
uv run metalbench eval-run \
  --kernelbench-dir KernelBench \
  --run-dir runs/metal_skills_v0_level1 \
  --correctness-trials 5 \
  --perf-trials 100 \
  --warmup 10 \
  --require-mps \
  --output runs/metal_skills_v0_level1/eval_results.json
```

Behavior:

- For each generated kernel, infer level/problem/sample from filename.
- Resolve reference files from the project-root `KernelBench/` directory using `kernelbench_adapter`.
- Evaluate each sample.
- Write JSON list to output.
- Continue after per-kernel failures.
- Include failure details in each result.

---

## 14. Phase 10: analysis

Implement `src/metalbench/analysis.py`.

Data model:

```python
class RunAnalysis(BaseModel):
    run_dir: Path
    total: int
    static_ok: int
    correct: int
    timed_generated: int
    timed_baseline: int
    metal_fast_0: float
    metal_fast_1: float
    metal_fast_2: float
    median_speedup_correct: float | None
    geomean_speedup_correct: float | None
    best_speedups: list[dict]
    worst_speedups: list[dict]
    failures_by_type: dict[str, int]
```

Function:

```python
def analyze_eval_results(eval_results_path: Path) -> RunAnalysis:
    ...
```

CLI:

```bash
uv run metalbench analyze runs/metal_skills_v0_level1/eval_results.json
uv run metalbench analyze runs/metal_skills_v0_level1/eval_results.json \
  --output runs/metal_skills_v0_level1/analysis.json
```

Analysis rules:

- `metal_fast_0` = correct / total.
- `metal_fast_1` = correct and speedup > 1 / total.
- `metal_fast_2` = correct and speedup > 2 / total.
- `median_speedup_correct` only over correct samples with valid baseline and generated timing.
- `geomean_speedup_correct` only over positive valid speedups.
- Failure buckets:
  - `static_check_failed`
  - `correctness_failed`
  - `generated_timing_failed`
  - `mps_baseline_failed`
  - `missing_reference`
  - `malformed_filename`

Tests:

- Use synthetic JSON fixtures.
- Verify metrics exactly.

---

## 15. Phase 11: generation contract

Implement `src/metalbench/generation_contract.py`.

Purpose:

- Define how external skills or Codex-generated kernels should be accepted.
- `metalbench` is not the LLM itself; it is the harness and contract verifier.

Data models:

```python
class GenerationRequest(BaseModel):
    level: int
    problem_id: int
    problem_name: str
    reference_source: str
    backend: str = "apple_metal"
    device: str = "mps"
    dtype: str = "fp32"
    requirements: list[str]


class GenerationArtifact(BaseModel):
    run_name: str
    level: int
    problem_id: int
    sample_id: int
    kernel_path: Path
    prompt_path: Path | None
    metadata_path: Path
```

Functions:

```python
def build_generation_prompt(problem: KBProblem) -> str:
    ...


def write_generation_request(
    problem: KBProblem,
    run_dir: Path,
    sample_id: int,
) -> GenerationArtifact:
    ...
```

CLI:

```bash
uv run metalbench generate-request \
  --kernelbench-dir KernelBench \
  --level 1 \
  --problem-id 19 \
  --run-dir runs/metal_skills_v0_level1 \
  --sample-id 0
```

This should write:

```text
runs/metal_skills_v0_level1/
  level_1_problem_19_sample_0_prompt.md
  level_1_problem_19_sample_0_metadata.json
```

The prompt must instruct the generator:

```text
Return a complete Python file.
Define class ModelNew(torch.nn.Module).
Use Apple Metal only.
Use torch.mps.compile_shader or torch.mps.load_metallib.
Do not use CUDA/Triton/HIP/ROCm.
Do not call the reference Model.
Do not use high-level PyTorch ops for the main computation.
Preserve the reference forward signature and semantics.
Assume inputs are on mps.
```

---

## 16. Phase 12: example generated kernels

Add `examples/references/synthetic_relu_reference.py`:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(1024, dtype=torch.float32)]
```

Add `examples/generated/level_1_problem_synthetic_relu_sample_0_kernel.py`:

```python
import torch
import torch.nn as nn

_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

kernel void relu_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        float v = x[gid];
        y[gid] = v > 0.0f ? v : 0.0f;
    }
}
"""

_LIB = torch.mps.compile_shader(_METAL_SRC)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.device.type == "mps"
        x = x.contiguous()
        y = torch.empty_like(x)
        _LIB.relu_f32(x, y, x.numel())
        return y
```

The static checker should accept this file.

---

## 17. Phase 13: CLI design

Implement `src/metalbench/cli.py` with Typer.

Commands:

```bash
uv run metalbench env
uv run metalbench kb list
uv run metalbench kb show
uv run metalbench check
uv run metalbench eval-one
uv run metalbench eval-run
uv run metalbench analyze
uv run metalbench generate-request
```

Examples:

```bash
uv run metalbench env --require-mps

uv run metalbench kb list \
  --kernelbench-dir KernelBench \
  --level 1

uv run metalbench generate-request \
  --kernelbench-dir KernelBench \
  --level 1 \
  --problem-id 19 \
  --run-dir runs/metal_skills_v0_level1 \
  --sample-id 0

uv run metalbench check \
  runs/metal_skills_v0_level1/level_1_problem_19_sample_0_kernel.py

uv run metalbench eval-one \
  --ref-path KernelBench/level1/19_*.py \
  --kernel-path runs/metal_skills_v0_level1/level_1_problem_19_sample_0_kernel.py \
  --require-mps

uv run metalbench eval-run \
  --kernelbench-dir KernelBench \
  --run-dir runs/metal_skills_v0_level1 \
  --require-mps \
  --output runs/metal_skills_v0_level1/eval_results.json

uv run metalbench analyze \
  runs/metal_skills_v0_level1/eval_results.json \
  --output runs/metal_skills_v0_level1/analysis.json
```

For shell glob paths in CLI examples, document that users may need to quote paths or use exact resolved reference paths.

---

## 18. Phase 14: repo-local Codex skill

Create `.agents/skills/metal-kernel-generation/SKILL.md`.

Skill file content:

```markdown
---
name: metal-kernel-generation
description: Generate and evaluate Apple Metal-only PyTorch ModelNew kernels for metalbench. Use this for KernelBench-to-Metal kernel generation, static validation, MPS correctness tests, and MPS timing. Never use CUDA, Triton, HIP, ROCm, or CPU fallback as an implementation path.
---

# Metal kernel generation rules

You are generating Apple Metal kernels for the `metalbench` project.

Hard requirements:
- Return complete Python files defining `class ModelNew(torch.nn.Module)`.
- Use Apple Metal only.
- Use `torch.mps.compile_shader(...)` or `torch.mps.load_metallib(...)`.
- Input tensors are expected on `torch.device("mps")`.
- Preserve the reference model's forward signature and semantics.
- Use CPU PyTorch only as a correctness oracle in the harness, never in generated kernels.
- Do not use CUDA, Triton, HIP, ROCm, CUTE, TileLang, ThunderKittens, CuPy, PyCUDA, nvcc, or CUDA C++.
- Do not call the reference `Model`.
- Do not use high-level PyTorch ops as the main computation path.
- Allocation helpers such as `torch.empty_like`, `torch.empty`, and `.contiguous()` are allowed.
- Prefer simple, correct Metal Shading Language first; optimize after correctness passes.

Generated file structure:

```python
import torch
import torch.nn as nn

_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

kernel void kernel_name(...) {
    ...
}
"""

_LIB = torch.mps.compile_shader(_METAL_SRC)

class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, ...):
        ...
```

Validation sequence:
1. Run `uv run metalbench check <generated_kernel.py>`.
2. Run `uv run metalbench eval-one ... --require-mps` on Apple Silicon.
3. Fix static-check failures before runtime debugging.
4. Fix correctness before optimizing performance.
5. Report speedup only against PyTorch MPS baseline.
```

---

## 19. Phase 15: README content

Create a concise `README.md` with these sections:

```text
# metalbench

Apple Metal-only kernel generation and evaluation harness for KernelBench-style PyTorch tasks.

## Scope
Included:
- Apple Silicon
- macOS
- PyTorch MPS
- Metal Shading Language
- torch.mps.compile_shader
- torch.mps.load_metallib
- CPU PyTorch correctness oracle
- MPS PyTorch performance baseline

Excluded:
- CUDA
- Triton
- HIP / ROCm
- NVIDIA / AMD benchmarking
- Modal
- CPU performance fallback

## Install

uv sync --all-extras

## KernelBench reference problems

`KernelBench/` must exist at the project root and must contain `level1/`, `level2/`, `level3/`, and `level4/` directly.

mkdir -p KernelBench
rsync -a /path/to/upstream/KernelBench/KernelBench/ ./KernelBench/

## Verify

uv run metalbench env
uv run metalbench env --require-mps

## Static check

uv run metalbench check examples/generated/level_1_problem_synthetic_relu_sample_0_kernel.py

## Evaluate synthetic example

uv run metalbench eval-one \
  --ref-path examples/references/synthetic_relu_reference.py \
  --kernel-path examples/generated/level_1_problem_synthetic_relu_sample_0_kernel.py \
  --require-mps

## Evaluate KernelBench run

uv run metalbench eval-run \
  --kernelbench-dir KernelBench \
  --run-dir runs/metal_skills_v0_level1 \
  --require-mps \
  --output runs/metal_skills_v0_level1/eval_results.json

uv run metalbench analyze \
  runs/metal_skills_v0_level1/eval_results.json \
  --output runs/metal_skills_v0_level1/analysis.json

## Metrics

metal_fast_0: fraction of generated kernels that are statically valid and correct.
metal_fast_1: fraction that are correct and faster than PyTorch MPS baseline.
metal_fast_2: fraction that are correct and at least 2x faster than PyTorch MPS baseline.
```

---

## 20. Phase 16: tests and acceptance gates

Non-MPS tests must pass anywhere:

```bash
uv run pytest \
  tests/test_env.py \
  tests/test_static_check.py \
  tests/test_import_utils.py \
  tests/test_analysis.py \
  tests/test_cli_smoke.py
```

MPS tests should be skipped unless Apple Metal is available:

```bash
uv run pytest -m mps
```

Full local Apple Silicon acceptance:

```bash
uv sync --all-extras
uv run metalbench env --require-mps
uv run metalbench check examples/generated/level_1_problem_synthetic_relu_sample_0_kernel.py
uv run metalbench eval-one \
  --ref-path examples/references/synthetic_relu_reference.py \
  --kernel-path examples/generated/level_1_problem_synthetic_relu_sample_0_kernel.py \
  --correctness-trials 5 \
  --perf-trials 100 \
  --warmup 10 \
  --require-mps
uv run pytest
uv run ruff check .
uv run mypy src
```


## 21. Phase 17: done definition

The `metalbench` project is done when:

```text
1. The project syncs with `uv sync --all-extras`.
2. `uv run metalbench --help` works.
3. `uv run metalbench env` works on any platform.
4. `uv run metalbench env --require-mps` fails clearly when MPS is unavailable.
5. Static checker accepts the synthetic Metal ReLU example.
6. Static checker rejects CUDA, Triton, HIP, missing ModelNew, and missing Metal entry point.
7. Synthetic MPS correctness/timing test passes on Apple Silicon.
8. Batch run layout supports `runs/<run_name>/level_<level>_problem_<problem_id>_sample_<sample_id>_kernel.py`.
9. `eval_results.json` and `analysis.json` are produced.
10. Metrics include `metal_fast_0`, `metal_fast_1`, `metal_fast_2`, and speedup versus PyTorch MPS baseline.
11. README documents the Metal-only scope and explicitly excludes CUDA/Triton/HIP.
12. `.agents/skills/metal-kernel-generation/SKILL.md` exists and enforces Metal-only generation.
```

---

## 22. First implementation sequence for Codex

Use this order to reduce failure risk:

```text
1. Create package skeleton and CLI.
2. Add or document the root-level `KernelBench/` reference-problem directory.
3. Implement env detection.
4. Implement import utilities.
5. Implement static checker.
6. Add synthetic reference and generated Metal ReLU example.
7. Add correctness utilities.
8. Add timing utilities.
9. Add eval-one.
10. Add analysis.
11. Add KernelBench local adapter that reads from root-level `./KernelBench/`.
12. Add eval-run.
13. Add generation request writer.
14. Add repo-local Codex skill.
15. Add README.
16. Run non-MPS tests with `uv run pytest`.
17. Run MPS tests with `uv run pytest -m mps` only if the current machine supports Apple Metal.
```

---

## 23. Reference notes

- Use `uv` as the project manager, dependency manager, lockfile manager, virtual environment manager, and command runner.
- `uv add` and `uv remove` should be the default mechanism for dependency changes; direct `pyproject.toml` edits are acceptable when Codex is generating files, but `uv lock` must be run afterward.
- `uv run` and `uv sync` should be the default execution and environment synchronization mechanisms; use `uv run --locked ...` for reproducible checks.
- KernelBench should be treated as a PyTorch problem/spec source and scoring inspiration, not as a Metal backend provider.
- The default KernelBench reference-problem location is `./KernelBench`, with files like `KernelBench/level1/19_*.py`; do not use `third_party/KernelBench` in this project plan.
- PyTorch MPS is the relevant runtime device for generated kernels.
- Generated kernels should use `torch.mps.compile_shader(...)` for inline Metal source or `torch.mps.load_metallib(...)` for precompiled Metal libraries.
- Do not report speedup against CPU or CUDA baselines.
- Do not fake benchmark results when running outside Apple Silicon/MPS environments.

The project-level invariant is:

> `metalbench` is a Metal/MPS-only harness; KernelBench supplies problem definitions, not GPU backend behavior.
