import json
from hashlib import sha256
from pathlib import Path

import pytest
from typer.testing import CliRunner

import metalbench.cli
import metalbench.env
from metalbench.cli import app
from metalbench.correctness import CorrectnessResult
from metalbench.eval_one import EvalOneResult
from metalbench.static_check import StaticCheckResult
from metalbench.timing import TimingResult


def test_cli_help_exits_successfully() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "metalbench" in result.output


def test_cli_env_exits_successfully_and_emits_json() -> None:
    result = CliRunner().invoke(app, ["env"])

    assert result.exit_code == 0
    environment = json.loads(result.output)
    assert environment["device"] == "mps"
    assert "cuda" not in result.output.lower()


def test_cli_env_require_mps_exits_with_error_when_mps_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)

    result = CliRunner().invoke(app, ["env", "--require-mps"])

    assert result.exit_code == 1
    assert "MPS" in result.output
    assert "cuda" not in result.output.lower()


def test_cli_check_valid_file_exits_successfully_and_emits_json(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.py"
    generated_path.write_text(
        """
import torch
import torch.nn as nn

_METAL_SRC = "kernel void noop() {}"
_LIB = torch.mps.compile_shader(_METAL_SRC)


class ModelNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        return y
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["check", str(generated_path)])

    assert result.exit_code == 0
    check_result = json.loads(result.output)
    assert check_result["ok"] is True
    assert check_result["errors"] == []
    assert check_result["path"] == str(generated_path)


def test_cli_check_invalid_file_exits_with_error_and_emits_json(tmp_path: Path) -> None:
    generated_path = tmp_path / "generated.py"
    generated_path.write_text(
        """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def forward(self, x):
        return torch.relu(x)
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["check", str(generated_path)])

    assert result.exit_code == 1
    check_result = json.loads(result.output)
    assert check_result["ok"] is False
    assert check_result["errors"]


def eval_one_result(ref_path: Path, kernel_path: Path) -> EvalOneResult:
    return EvalOneResult(
        run_name="run",
        level=1,
        problem_id=19,
        sample_id=0,
        ref_path=ref_path,
        kernel_path=kernel_path,
        static_check=StaticCheckResult(
            path=kernel_path,
            ok=True,
            errors=[],
            warnings=[],
            uses_compile_shader=True,
            uses_load_metallib=False,
            found_metal_kernel_source=True,
        ),
        correctness=CorrectnessResult(ok=True, trials=1, passed=1, failed=0, errors=[]),
        generated_timing=TimingResult(
            ok=True,
            device="mps",
            median_ms=2.0,
            mean_ms=2.0,
            min_ms=2.0,
            max_ms=2.0,
            p25_ms=2.0,
            p75_ms=2.0,
            warmup=1,
            trials=1,
            errors=[],
        ),
        mps_baseline_timing=TimingResult(
            ok=True,
            device="mps",
            median_ms=5.0,
            mean_ms=5.0,
            min_ms=5.0,
            max_ms=5.0,
            p25_ms=5.0,
            p75_ms=5.0,
            warmup=1,
            trials=1,
            errors=[],
        ),
        speedup_vs_mps=2.5,
        metal_fast_0=True,
        metal_fast_1=True,
        metal_fast_2=True,
        errors=[],
    )


def test_cli_eval_one_emits_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    monkeypatch.setattr(
        metalbench.cli.eval_one,
        "evaluate_one",
        lambda *args, **kwargs: eval_one_result(ref_path, kernel_path),
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-one",
            "--ref-path",
            str(ref_path),
            "--kernel-path",
            str(kernel_path),
        ],
    )

    assert result.exit_code == 0
    eval_result = json.loads(result.output)
    assert eval_result["ref_path"] == str(ref_path)
    assert eval_result["kernel_path"] == str(kernel_path)
    assert eval_result["speedup_vs_mps"] == 2.5
    assert eval_result["metal_fast_2"] is True


def test_cli_eval_one_output_writes_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    output_path = tmp_path / "result.json"
    monkeypatch.setattr(
        metalbench.cli.eval_one,
        "evaluate_one",
        lambda *args, **kwargs: eval_one_result(ref_path, kernel_path),
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-one",
            "--ref-path",
            str(ref_path),
            "--kernel-path",
            str(kernel_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert result.output == ""
    eval_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert eval_result["ref_path"] == str(ref_path)
    assert eval_result["speedup_vs_mps"] == 2.5


def test_cli_eval_one_require_mps_exits_with_error_when_mps_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)

    result = CliRunner().invoke(
        app,
        [
            "eval-one",
            "--ref-path",
            str(tmp_path / "reference.py"),
            "--kernel-path",
            str(tmp_path / "kernel.py"),
            "--require-mps",
        ],
    )

    assert result.exit_code == 1
    assert "MPS" in result.output


def test_cli_eval_run_output_writes_json_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "level_1_problem_19_sample_0_kernel.py"
    output_path = tmp_path / "results.json"
    monkeypatch.setattr(
        metalbench.cli.eval_batch,
        "evaluate_run_directory",
        lambda *args, **kwargs: [eval_one_result(ref_path, kernel_path)],
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-run",
            "--run-dir",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert result.output == ""
    eval_results = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(eval_results, list)
    assert eval_results[0]["ref_path"] == str(ref_path)
    assert eval_results[0]["kernel_path"] == str(kernel_path)


def test_cli_eval_run_require_mps_exits_with_error_when_mps_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)

    result = CliRunner().invoke(
        app,
        [
            "eval-run",
            "--run-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "results.json"),
            "--require-mps",
        ],
    )

    assert result.exit_code == 1
    assert "MPS" in result.output


def test_cli_eval_run_passes_options_to_evaluate_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results.json"
    kernelbench_dir = tmp_path / "KernelBench"
    calls: list[dict[str, object]] = []

    def fake_evaluate_run_directory(
        run_dir: Path,
        kernelbench_dir: Path,
        **kwargs: object,
    ) -> list[EvalOneResult]:
        calls.append(
            {
                "run_dir": run_dir,
                "kernelbench_dir": kernelbench_dir,
                **kwargs,
            }
        )
        return []

    monkeypatch.setattr(
        metalbench.cli.eval_batch,
        "evaluate_run_directory",
        fake_evaluate_run_directory,
    )

    result = CliRunner().invoke(
        app,
        [
            "eval-run",
            "--run-dir",
            str(tmp_path),
            "--kernelbench-dir",
            str(kernelbench_dir),
            "--correctness-trials",
            "7",
            "--perf-trials",
            "11",
            "--warmup",
            "13",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == []
    assert calls == [
        {
            "run_dir": tmp_path,
            "kernelbench_dir": kernelbench_dir,
            "correctness_trials": 7,
            "perf_trials": 11,
            "warmup": 13,
            "require_mps": False,
        }
    ]


def test_cli_generate_request_writes_prompt_and_metadata(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    source = "import torch\n\nclass Model:\n    pass\n"
    write_problem(kernelbench_dir, source)
    run_dir = tmp_path / "run"

    result = CliRunner().invoke(
        app,
        [
            "generate-request",
            "--kernelbench-dir",
            str(kernelbench_dir),
            "--level",
            "1",
            "--problem-id",
            "19",
            "--run-dir",
            str(run_dir),
            "--sample-id",
            "2",
        ],
    )

    assert result.exit_code == 0
    artifact = json.loads(result.output)
    assert artifact["run_name"] == "run"
    assert artifact["level"] == 1
    assert artifact["problem_id"] == 19
    assert artifact["sample_id"] == 2
    assert artifact["prompt_path"] == str(run_dir / "level_1_problem_19_sample_2_prompt.md")
    assert artifact["metadata_path"] == str(run_dir / "level_1_problem_19_sample_2_metadata.json")
    assert artifact["kernel_path"] == str(run_dir / "level_1_problem_19_sample_2_kernel.py")
    assert Path(artifact["prompt_path"]).is_file()
    assert Path(artifact["metadata_path"]).is_file()


def test_cli_generate_request_missing_kernelbench_dir_exits_with_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "generate-request",
            "--kernelbench-dir",
            str(tmp_path / "missing"),
            "--level",
            "1",
            "--problem-id",
            "19",
            "--run-dir",
            str(tmp_path / "run"),
            "--sample-id",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "KernelBench directory not found" in result.output


def test_cli_analyze_emits_json(tmp_path: Path) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    results_path = tmp_path / "eval_results.json"
    results_path.write_text(
        json.dumps([eval_one_result(ref_path, kernel_path).model_dump(mode="json")]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["analyze", str(results_path)])

    assert result.exit_code == 0
    analysis = json.loads(result.output)
    assert analysis["run_dir"] == str(tmp_path)
    assert analysis["total"] == 1
    assert analysis["metal_fast_2"] == 1.0


def test_cli_analyze_output_writes_json_and_prints_nothing(tmp_path: Path) -> None:
    ref_path = tmp_path / "reference.py"
    kernel_path = tmp_path / "kernel.py"
    results_path = tmp_path / "eval_results.json"
    output_path = tmp_path / "analysis.json"
    results_path.write_text(
        json.dumps([eval_one_result(ref_path, kernel_path).model_dump(mode="json")]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["analyze", str(results_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert result.output == ""
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    analysis = json.loads(output_path.read_text(encoding="utf-8"))
    assert analysis["total"] == 1


def test_cli_analyze_invalid_input_exits_with_error(tmp_path: Path) -> None:
    results_path = tmp_path / "eval_results.json"
    results_path.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(app, ["analyze", str(results_path)])

    assert result.exit_code == 1
    assert "JSON list" in result.output


def write_problem(kernelbench_dir: Path, source: str = "class Model:\n    pass\n") -> Path:
    level_dir = kernelbench_dir / "level1"
    level_dir.mkdir(parents=True)
    path = level_dir / "19_ReLU.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_cli_kb_list_emits_json_without_source(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    source = "class Model:\n    pass\n"
    path = write_problem(kernelbench_dir, source)

    result = CliRunner().invoke(
        app,
        ["kb", "list", "--kernelbench-dir", str(kernelbench_dir), "--level", "1"],
    )

    assert result.exit_code == 0
    problems = json.loads(result.output)
    assert problems == [
        {
            "level": 1,
            "problem_id": 19,
            "name": "ReLU",
            "path": str(path),
            "source_sha256": sha256(source.encode("utf-8")).hexdigest(),
        }
    ]
    assert "source" not in problems[0]


def test_cli_kb_list_missing_directory_exits_with_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["kb", "list", "--kernelbench-dir", str(tmp_path / "missing"), "--level", "1"],
    )

    assert result.exit_code == 1
    assert "KernelBench directory not found" in result.output


def test_cli_kb_show_emits_json_without_source_by_default(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    path = write_problem(kernelbench_dir)

    result = CliRunner().invoke(
        app,
        [
            "kb",
            "show",
            "--kernelbench-dir",
            str(kernelbench_dir),
            "--level",
            "1",
            "--problem-id",
            "19",
        ],
    )

    assert result.exit_code == 0
    problem = json.loads(result.output)
    assert problem["name"] == "ReLU"
    assert problem["path"] == str(path)
    assert "source" not in problem


def test_cli_kb_show_print_source_includes_source(tmp_path: Path) -> None:
    kernelbench_dir = tmp_path / "KernelBench"
    source = "class Model:\n    pass\n"
    write_problem(kernelbench_dir, source)

    result = CliRunner().invoke(
        app,
        [
            "kb",
            "show",
            "--kernelbench-dir",
            str(kernelbench_dir),
            "--level",
            "1",
            "--problem-id",
            "19",
            "--print-source",
        ],
    )

    assert result.exit_code == 0
    problem = json.loads(result.output)
    assert problem["source"] == source
