import json
from pathlib import Path

from metalbench.generation_contract import build_generation_prompt, write_generation_request
from metalbench.types import KBProblem


def kb_problem(path: Path) -> KBProblem:
    return KBProblem(
        level=1,
        problem_id=19,
        name="ReLU",
        path=path,
        source="import torch\n\nclass Model:\n    pass\n",
        source_sha256="abc123",
    )


def test_build_generation_prompt_includes_metal_contract_and_reference_source(
    tmp_path: Path,
) -> None:
    problem = kb_problem(tmp_path / "19_ReLU.py")

    prompt = build_generation_prompt(problem)

    assert "class ModelNew" in prompt
    assert "Apple Metal only" in prompt
    assert "torch.mps.compile_shader or torch.mps.load_metallib" in prompt
    assert "Do not use CUDA, Triton, HIP, ROCm, CuPy, PyCUDA, nvcc, or .cu files." in prompt
    assert "Assume all inputs are already on the MPS device." in prompt
    assert problem.source in prompt


def test_write_generation_request_creates_prompt_and_metadata_without_kernel(
    tmp_path: Path,
) -> None:
    problem = kb_problem(tmp_path / "19_ReLU.py")
    run_dir = tmp_path / "run"

    artifact = write_generation_request(problem, run_dir, sample_id=3)

    assert run_dir.is_dir()
    assert artifact.run_name == "run"
    assert artifact.level == 1
    assert artifact.problem_id == 19
    assert artifact.sample_id == 3
    assert artifact.prompt_path == run_dir / "level_1_problem_19_sample_3_prompt.md"
    assert artifact.metadata_path == run_dir / "level_1_problem_19_sample_3_metadata.json"
    assert artifact.kernel_path == run_dir / "level_1_problem_19_sample_3_kernel.py"
    assert not artifact.kernel_path.exists()
    assert artifact.prompt_path.read_text(encoding="utf-8") == build_generation_prompt(problem)

    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert sorted(metadata) == ["artifact", "request"]
    assert metadata["request"]["level"] == 1
    assert metadata["request"]["problem_id"] == 19
    assert metadata["request"]["problem_name"] == "ReLU"
    assert metadata["request"]["reference_source"] == problem.source
    assert metadata["request"]["backend"] == "apple_metal"
    assert metadata["request"]["device"] == "mps"
    assert metadata["request"]["dtype"] == "fp32"
    assert metadata["artifact"] == artifact.model_dump(mode="json")
