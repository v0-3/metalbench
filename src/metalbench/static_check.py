from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class StaticCheckResult(BaseModel):
    path: Path
    ok: bool
    errors: list[str]
    warnings: list[str]
    uses_compile_shader: bool
    uses_load_metallib: bool
    found_metal_kernel_source: bool


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class TokenRuleGroup:
    name: str
    severity: Severity
    tokens: tuple[str, ...]


BLOCKED_BACKEND_TOKENS = TokenRuleGroup(
    name="blocked backend token",
    severity="error",
    tokens=(
        "torch.cuda",
        "cuda",
        "CUDA",
        "triton",
        "tl.",
        "hip",
        "rocm",
        "ROCm",
        "cupy",
        "pycuda",
        "nvcc",
        "__global__",
        "__device__",
        "__host__",
        ".cu",
    ),
)
BLOCKED_REFERENCE_TOKENS = TokenRuleGroup(
    name="blocked reference or fallback token",
    severity="error",
    tokens=(
        "class Model(",
        "Model(",
        "ref_model",
        "reference_model",
        "torch.nn.functional.relu",
        "torch.relu",
        "torch.matmul",
        "torch.mm",
        "torch.bmm",
        "torch.conv",
        "torch.nn.functional.conv",
    ),
)
WARNING_TOKENS = TokenRuleGroup(
    name="warning token",
    severity="warning",
    tokens=(
        ".cpu()",
        ".numpy()",
    ),
)
TOKEN_RULE_GROUPS = (
    BLOCKED_BACKEND_TOKENS,
    BLOCKED_REFERENCE_TOKENS,
    WARNING_TOKENS,
)
ALLOCATION_HELPER_TOKENS = (
    "torch.empty_like",
    "torch.empty",
    ".contiguous()",
)


def check_generated_metal_kernel(path: Path) -> StaticCheckResult:
    if not path.exists():
        raise FileNotFoundError(f"Generated kernel path not found: {path}")
    if not path.is_file():
        raise ValueError(f"Generated kernel path is not a file: {path}")

    source = path.read_text(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []

    for group in TOKEN_RULE_GROUPS:
        _apply_token_group(group, source, errors, warnings)

    uses_compile_shader = "torch.mps.compile_shader" in source
    uses_load_metallib = "torch.mps.load_metallib" in source
    found_metal_kernel_source = "kernel void" in source

    if "class ModelNew" not in source:
        errors.append("Missing required token: class ModelNew")
    if not uses_compile_shader and not uses_load_metallib:
        errors.append(
            "Missing required Metal loader: torch.mps.compile_shader or torch.mps.load_metallib"
        )
    if uses_compile_shader and not found_metal_kernel_source:
        errors.append("Missing visible Metal kernel source token: kernel void")
    if not any(token in source for token in ALLOCATION_HELPER_TOKENS):
        warnings.append(
            "Missing common allocation/helper token: "
            "torch.empty_like, torch.empty, or .contiguous()"
        )
    if _contains_forward_synchronize(source):
        warnings.append("torch.mps.synchronize() appears inside forward")

    return StaticCheckResult(
        path=path,
        ok=not errors,
        errors=errors,
        warnings=warnings,
        uses_compile_shader=uses_compile_shader,
        uses_load_metallib=uses_load_metallib,
        found_metal_kernel_source=found_metal_kernel_source,
    )


def _apply_token_group(
    group: TokenRuleGroup,
    source: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    messages = errors if group.severity == "error" else warnings
    for token in group.tokens:
        if token in source:
            messages.append(f"{group.name}: {token}")


def _contains_forward_synchronize(source: str) -> bool:
    forward_indent: int | None = None

    for line in source.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue

        indent = len(line) - len(stripped)
        if forward_indent is None:
            if stripped.startswith("def forward("):
                forward_indent = indent
            continue

        if indent <= forward_indent and not stripped.startswith("#"):
            forward_indent = None
            if stripped.startswith("def forward("):
                forward_indent = indent
            continue

        if "torch.mps.synchronize()" in stripped:
            return True

    return False


__all__ = [
    "StaticCheckResult",
    "check_generated_metal_kernel",
]
