"""Exercise noneditable artifacts without checkout or editable-import fallback."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(args, *, cwd, env):
    result = subprocess.run(args, cwd=cwd, env=env, text=True,
                            capture_output=True, timeout=120)
    (Path(cwd) / "subprocess.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("noneditable-wheel")
    source, wheels, installed = (scratch / name for name in ("source", "wheels", "installed"))
    source.mkdir()
    wheels.mkdir()
    # Build tracked source in scratch, including current tracked-file edits. Never
    # reuse a checkout's build directory, egg-info, or editable installation.
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for name in filter(None, tracked):
        original = ROOT / name
        if original.is_file():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    env = {"PATH": os.defpath, "PIP_NO_INDEX": "1",
           "NEUROFLY_BRAIN_BACKEND": "cpu"}
    _run([sys.executable, "-I", "-c",
          "import setuptools.build_meta as b; import sys; b.build_wheel(sys.argv[1])",
          str(wheels)], cwd=source, env=env)
    (scratch / "build.log").write_text((source / "subprocess.log").read_text())
    wheel, = wheels.glob("*.whl")
    _run([sys.executable, "-I", "-m", "pip", "install", "--no-index", "--no-deps",
          "--target", str(installed), str(wheel)], cwd=scratch, env=env)
    (scratch / "install.log").write_text((scratch / "subprocess.log").read_text())
    # -S prevents site processing: dependency directories are added explicitly,
    # so existing editable .pth files and their import finders are never loaded.
    dependencies = sorted({sysconfig.get_path(key) for key in ("purelib", "platlib")})
    local_names = sorted({p.stem for p in ROOT.glob("*.py")} |
                         {p.parent.name for p in ROOT.glob("*/__init__.py")})
    return scratch, installed, dependencies, local_names, env


@pytest.mark.parametrize("entrypoint", ["daemon", "cli_help"])
def test_installed_entrypoints_resolve_transitive_local_imports_from_wheel(installed_wheel, entrypoint):
    scratch, installed, dependencies, local_names, env = installed_wheel
    bootstrap = r'''
import importlib, json, pathlib, runpy, sys
installed = pathlib.Path(sys.argv[1]).resolve()
dependencies, local_names = json.loads(sys.argv[2]), set(json.loads(sys.argv[3]))
sys.path[:] = [str(installed)] + dependencies + [p for p in sys.path if p]
if sys.argv[4] == "daemon":
    daemon = importlib.import_module("neurofly_daemon")
    assert daemon.PROJECT_ROOT == installed
    assert daemon.WEB_BUILD_PLACEHOLDER.encode() not in daemon.dashboard_index_bytes(installed / "web/index.html")
else:
    sys.argv[:] = ["neurofly", "--help"]
    try:
        runpy.run_path(str(installed / "bin" / "neurofly"), run_name="__main__")
    except SystemExit as exc:
        assert exc.code in (None, 0), exc.code
origins = {}
for name, module in tuple(sys.modules.items()):
    if name.split(".")[0] in local_names and getattr(module, "__file__", None):
        path = pathlib.Path(module.__file__).resolve()
        assert path.is_relative_to(installed), (name, str(path))
        origins[name] = str(path)
assert origins, "No installed project module was exercised"
assert not any(name == "cupy" or name.startswith("numba.cuda") for name in sys.modules)
print("WHEEL_IMPORT_ORIGINS=" + json.dumps(origins, sort_keys=True))
'''
    output = _run([sys.executable, "-I", "-S", "-c", bootstrap, str(installed),
                   json.dumps(dependencies), json.dumps(local_names), entrypoint],
                  cwd=scratch, env=env)
    (scratch / (entrypoint + ".log")).write_text(output)
    assert "WHEEL_IMPORT_ORIGINS=" in output
    if entrypoint == "cli_help":
        assert "usage:" in output.lower()


def test_wheel_contains_all_dashboard_assets(installed_wheel):
    _, installed, _, _, _ = installed_wheel
    # Include linked pages and JS/CSS as well as current build-hash inputs.
    # Compare against the actual frontend tree, not a copied filename allowlist.
    assets = [path for path in (ROOT / "web").rglob("*")
              if path.is_file() and path.suffix in {".html", ".js", ".css"}]
    assert assets
    for asset in assets:
        packaged = installed / asset.relative_to(ROOT)
        assert packaged.is_file(), f"Wheel omits {asset.relative_to(ROOT)}"
        assert packaged.read_bytes() == asset.read_bytes()


def test_wheel_contains_frozen_amd_shader_optional_dependency_and_lazy_import(installed_wheel):
    scratch, installed, dependencies, _, env = installed_wheel
    shader = ROOT / 'brainlab/wgpu_v3_math.wgsl'
    assert (installed / 'brainlab/wgpu_v3_math.wgsl').read_bytes() == shader.read_bytes()
    metadata, = installed.glob('neurofly-*.dist-info/METADATA')
    assert 'Requires-Dist: wgpu==0.32.0; extra == "amd"' in metadata.read_text()
    bootstrap = r'''
import importlib, json, pathlib, sys, hashlib
installed = pathlib.Path(sys.argv[1]).resolve()
sys.path[:] = [str(installed)] + json.loads(sys.argv[2]) + [p for p in sys.path if p]
for name in ('neurofly_daemon', 'brainlab.brain', 'brainlab.runtime_backend',
             'brainlab.amd_state_adapter', 'brainlab.wgpu_v3'):
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(installed), (name, module.__file__)
from brainlab import wgpu_v3
assert hashlib.sha256(wgpu_v3.MATH_SOURCE.encode()).hexdigest() == 'a4f666380fc4ccb4b2737462a475484b0c37f14e71470bfb6a4d28355f8e494e'
assert not any(n == 'wgpu' or n.startswith('wgpu.') for n in sys.modules)
print('WHEEL_AMD_LAZY_IMPORT_OK')
'''
    output = _run([sys.executable, '-I', '-S', '-c', bootstrap, str(installed),
                   json.dumps(dependencies)], cwd=scratch, env=env)
    assert 'WHEEL_AMD_LAZY_IMPORT_OK' in output
