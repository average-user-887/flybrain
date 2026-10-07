"""F2 packaging: an installed wheel resolves the FlyWire source lock and pins from its own
package, in a neutral working directory, and ships no annotation TSV or derived table."""
import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DATA_SUFFIXES = (".tsv", ".feather", ".npz", ".npy", ".parquet", ".csv")


def test_packaged_lock_matches_the_documented_lock():
    from brainlab import flywire

    assert flywire.LOCK_PATH.parent == Path(flywire.__file__).resolve().parent
    assert flywire.LOCK_PATH.read_bytes() == flywire.REPO_LOCK_PATH.read_bytes()


def _source_copy(dest: Path) -> Path:
    """The files a wheel build reads, copied without repository data directories."""
    config = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        if (REPO / name).exists():
            shutil.copy2(REPO / name, dest / name)
    for module in config["py-modules"]:
        shutil.copy2(REPO / f"{module}.py", dest / f"{module}.py")
    mapped = config.get("package-dir", {})
    for package in config["packages"]:
        rel = mapped.get(package, package.replace(".", "/"))
        src, out = REPO / rel, dest / rel
        out.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, out / item.name)
        for sub in ("vendor", "specs", "specs/sources"):
            if (src / sub).is_dir() and not (out / sub).exists():
                shutil.copytree(src / sub, out / sub)
    return dest


def test_installed_wheel_loads_the_lock_from_a_neutral_cwd(tmp_path):
    (tmp_path / "src").mkdir()
    src = _source_copy(tmp_path / "src")
    wheels = tmp_path / "wheels"
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    "--no-cache-dir", "-q", "-w", str(wheels), str(src)], check=True,
                   env={**os.environ, "SOURCE_DATE_EPOCH": "0"})
    wheel = next(wheels.glob("neurofly-*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    assert "brainlab/source_lock_flywire_783_female.json" in names
    assert "brainlab/graph_pins_flywire_783_female.json" in names
    assert not [n for n in names if n.lower().endswith(DATA_SUFFIXES)], "data tables must not be packaged"
    assert not [n for n in names if "annotation" in n.lower()]
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    code = ("import json, brainlab.flywire as f; lock = f.load_lock(); pins = f.load_pins(); "
            "print(json.dumps({'file': f.__file__, 'lock': str(f.LOCK_PATH), 'files': sorted(lock['files']), "
            "'pins': pins['dataset_id']}))")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(site)
    out = subprocess.run([sys.executable, "-c", code], cwd=neutral, env=env, check=True,
                         capture_output=True, text=True).stdout
    result = json.loads(out)
    assert result["file"].startswith(str(site)) and result["lock"].startswith(str(site))
    assert result["pins"] == "flywire_783_female"
    assert "proofread_connections_783.feather" in result["files"]
