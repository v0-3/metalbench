import json
from pathlib import Path

from pydantic import BaseModel

from metalbench.types import KBProblem

GENERATION_REQUIREMENTS = (
    "Return a Python file that defines class ModelNew.",
    "Use Apple Metal only for custom kernels.",
    "Use torch.mps.compile_shader or torch.mps.load_metallib to load Metal kernels.",
    "If using torch.mps.compile_shader, include visible Metal source with a "
    "kernel void entry point.",
    "Do not use CUDA, Triton, HIP, ROCm, CuPy, PyCUDA, nvcc, or .cu files.",
    "Do not call the reference Model, ref_model, reference_model, or blocked "
    "torch reference ops in forward.",
    "Assume all inputs are already on the MPS device.",
    "Use fp32 outputs unless the reference problem requires another dtype.",
    "Use torch.empty, torch.empty_like, or .contiguous() where allocation or "
    "layout handling is needed.",
    "Do not call torch.mps.synchronize() inside forward.",
)


class GenerationRequest(BaseModel):
    level: int
    problem_id: int
    problem_name: str
    reference_source: str
    backend: str = "apple_metal"
    device: str = "mps"
    dtype: str = "fp32"
    requirements: tuple[str, ...] = GENERATION_REQUIREMENTS


class GenerationArtifact(BaseModel):
    run_name: str
    level: int
    problem_id: int
    sample_id: int
    kernel_path: Path
    prompt_path: Path
    metadata_path: Path


def build_generation_prompt(problem: KBProblem) -> str:
    request = _generation_request(problem)
    requirements = "\n".join(f"- {requirement}" for requirement in request.requirements)
    return (
        "# MetalBench Generation Request\n\n"
        "Generate an Apple Metal kernel implementation for this KernelBench problem.\n\n"
        "## Target\n\n"
        f"- Level: {request.level}\n"
        f"- Problem ID: {request.problem_id}\n"
        f"- Problem name: {request.problem_name}\n"
        f"- Backend: {request.backend}\n"
        f"- Device: {request.device}\n"
        f"- Dtype: {request.dtype}\n\n"
        "## Requirements\n\n"
        f"{requirements}\n\n"
        "## Reference Source\n\n"
        "```python\n"
        f"{request.reference_source}"
        "```\n"
    )


def write_generation_request(
    problem: KBProblem,
    run_dir: Path,
    sample_id: int,
) -> GenerationArtifact:
    run_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"level_{problem.level}_problem_{problem.problem_id}_sample_{sample_id}"
    artifact = GenerationArtifact(
        run_name=run_dir.name,
        level=problem.level,
        problem_id=problem.problem_id,
        sample_id=sample_id,
        kernel_path=run_dir / f"{prefix}_kernel.py",
        prompt_path=run_dir / f"{prefix}_prompt.md",
        metadata_path=run_dir / f"{prefix}_metadata.json",
    )
    request = _generation_request(problem)

    artifact.prompt_path.write_text(build_generation_prompt(problem), encoding="utf-8")
    metadata = {
        "request": request.model_dump(mode="json"),
        "artifact": artifact.model_dump(mode="json"),
    }
    artifact.metadata_path.write_text(
        f"{json.dumps(metadata, indent=2)}\n",
        encoding="utf-8",
    )

    return artifact


def _generation_request(problem: KBProblem) -> GenerationRequest:
    return GenerationRequest(
        level=problem.level,
        problem_id=problem.problem_id,
        problem_name=problem.name,
        reference_source=problem.source,
    )


__all__ = [
    "GenerationArtifact",
    "GenerationRequest",
    "build_generation_prompt",
    "write_generation_request",
]
