from pathlib import Path

from pydantic import BaseModel


class KBProblem(BaseModel):
    level: int
    problem_id: int
    name: str
    path: Path
    source: str
    source_sha256: str


__all__ = ["KBProblem"]
