from pathlib import Path

import pytest

from metalbench.static_check import check_generated_metal_kernel


def write_generated(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def minimal_compile_shader_source(extra_forward: str = "") -> str:
    return f"""
import torch
import torch.nn as nn

_METAL_SRC = \"\"\"
kernel void relu_f32() {{
}}
\"\"\"

_LIB = torch.mps.compile_shader(_METAL_SRC)


class ModelNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        {extra_forward}
        return y
"""


def minimal_load_metallib_source() -> str:
    return """
import torch
import torch.nn as nn

_LIB = torch.mps.load_metallib("kernel.metallib")


class ModelNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        return y
"""


def test_accepts_minimal_compile_shader_sample(tmp_path: Path) -> None:
    path = write_generated(tmp_path / "generated.py", minimal_compile_shader_source())

    result = check_generated_metal_kernel(path)

    assert result.path == path
    assert result.ok is True
    assert result.errors == []
    assert result.uses_compile_shader is True
    assert result.uses_load_metallib is False
    assert result.found_metal_kernel_source is True


def test_accepts_minimal_load_metallib_sample(tmp_path: Path) -> None:
    path = write_generated(tmp_path / "generated.py", minimal_load_metallib_source())

    result = check_generated_metal_kernel(path)

    assert result.ok is True
    assert result.errors == []
    assert result.uses_compile_shader is False
    assert result.uses_load_metallib is True
    assert result.found_metal_kernel_source is False


@pytest.mark.parametrize("token", ["torch.cuda", "CUDA", "__global__", ".cu"])
def test_rejects_cuda_token_usage(tmp_path: Path, token: str) -> None:
    path = write_generated(
        tmp_path / "generated.py",
        minimal_compile_shader_source(extra_forward=f"# forbidden backend: {token}"),
    )

    result = check_generated_metal_kernel(path)

    assert result.ok is False
    assert any(token in error for error in result.errors)


@pytest.mark.parametrize("token", ["triton", "tl."])
def test_rejects_triton_token_usage(tmp_path: Path, token: str) -> None:
    path = write_generated(
        tmp_path / "generated.py",
        minimal_compile_shader_source(extra_forward=f"# forbidden backend: {token}"),
    )

    result = check_generated_metal_kernel(path)

    assert result.ok is False
    assert any(token in error for error in result.errors)


def test_rejects_missing_modelnew(tmp_path: Path) -> None:
    path = write_generated(
        tmp_path / "generated.py",
        minimal_compile_shader_source().replace("class ModelNew", "class OtherModel"),
    )

    result = check_generated_metal_kernel(path)

    assert result.ok is False
    assert any("class ModelNew" in error for error in result.errors)


def test_rejects_missing_metal_entry_point_for_compile_shader(tmp_path: Path) -> None:
    path = write_generated(
        tmp_path / "generated.py",
        minimal_compile_shader_source().replace("kernel void", "void"),
    )

    result = check_generated_metal_kernel(path)

    assert result.ok is False
    assert any("kernel void" in error for error in result.errors)
    assert result.found_metal_kernel_source is False


@pytest.mark.parametrize("token", [".cpu()", ".numpy()"])
def test_warns_on_host_transfer_calls(tmp_path: Path, token: str) -> None:
    path = write_generated(
        tmp_path / "generated.py",
        minimal_compile_shader_source(extra_forward=f"x{token}"),
    )

    result = check_generated_metal_kernel(path)

    assert result.ok is True
    assert any(token in warning for warning in result.warnings)


def test_warns_on_mps_synchronize_inside_forward(tmp_path: Path) -> None:
    path = write_generated(
        tmp_path / "generated.py",
        minimal_compile_shader_source(extra_forward="torch.mps.synchronize()"),
    )

    result = check_generated_metal_kernel(path)

    assert result.ok is True
    assert any("torch.mps.synchronize()" in warning for warning in result.warnings)


def test_warns_when_allocation_helper_tokens_are_absent(tmp_path: Path) -> None:
    source = """
import torch
import torch.nn as nn

_METAL_SRC = "kernel void noop() {}"
_LIB = torch.mps.compile_shader(_METAL_SRC)


class ModelNew(nn.Module):
    def forward(self, x):
        return x
"""
    path = write_generated(tmp_path / "generated.py", source)

    result = check_generated_metal_kernel(path)

    assert result.ok is True
    assert any("torch.empty_like" in warning for warning in result.warnings)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.py"

    with pytest.raises(FileNotFoundError, match="Generated kernel path not found"):
        check_generated_metal_kernel(missing_path)


def test_directory_path_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Generated kernel path is not a file"):
        check_generated_metal_kernel(tmp_path)
