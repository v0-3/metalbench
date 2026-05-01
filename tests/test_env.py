from importlib.metadata import metadata, version

import pytest

import metalbench
import metalbench.env

EXPECTED_ENV_KEYS = {
    "python",
    "platform",
    "torch_version",
    "mps_available",
    "mps_built",
    "device",
}


def test_installed_package_metadata_matches_imported_version() -> None:
    package_metadata = metadata("metalbench")

    assert package_metadata["Name"] == "metalbench"
    assert package_metadata["Summary"]
    assert version("metalbench") == metalbench.__version__ == "0.1.0"


def test_describe_environment_has_expected_keys() -> None:
    environment = metalbench.env.describe_environment()

    assert set(environment) == EXPECTED_ENV_KEYS


def test_describe_environment_has_stable_value_types() -> None:
    environment = metalbench.env.describe_environment()

    assert isinstance(environment["python"], str)
    assert isinstance(environment["platform"], str)
    assert isinstance(environment["torch_version"], str)
    assert isinstance(environment["mps_available"], bool)
    assert isinstance(environment["mps_built"], bool)
    assert isinstance(environment["device"], str)


def test_describe_environment_reports_mps_device() -> None:
    environment = metalbench.env.describe_environment()

    assert environment["device"] == "mps"


def test_is_mps_available_returns_bool() -> None:
    assert isinstance(metalbench.env.is_mps_available(), bool)


def test_require_mps_succeeds_when_mps_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: True)

    metalbench.env.require_mps()


def test_require_mps_raises_when_mps_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metalbench.env, "is_mps_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS"):
        metalbench.env.require_mps()
