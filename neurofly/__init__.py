"""Project NeuroFly — Whole-Brain Drosophila Connectome Coupled to Embodied Biomechanics."""
from pathlib import Path as _Path


def _package_version() -> str:
    """The version declared in pyproject.toml (source checkout), else the installed metadata.

    pyproject.toml is read first because an editable install keeps the metadata of
    the version it was installed at.
    """
    pyproject = _Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        import tomllib
        with pyproject.open("rb") as handle:
            return tomllib.load(handle)["project"]["version"]
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("neurofly")
    except Exception:
        return "0+unknown"


__version__ = _package_version()
