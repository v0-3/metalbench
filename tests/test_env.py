from importlib.metadata import metadata, version

import metalbench


def test_installed_package_metadata_matches_imported_version() -> None:
    package_metadata = metadata("metalbench")

    assert package_metadata["Name"] == "metalbench"
    assert package_metadata["Summary"]
    assert version("metalbench") == metalbench.__version__ == "0.1.0"
