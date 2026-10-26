"""Machine-specific locations, so the code runs on any machine without editing paths.

Everything inside the repo is found from this file's location. Only data that lives elsewhere is configured here, with this
priority: environment variable, then `code/config.local.toml` (not committed; copy `config.example.toml`), then the default.

    GDELT_DIR   GDELT query cache, ledger, pulled data and explorer files   (default: <repo>/data/gdelt)
    PLOTS_DIR   figures and tables written by the plotting scripts           (default: <repo>/gdelt_plots)
"""
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:       # Python < 3.11
    tomllib = None

ROOT = Path(__file__).resolve().parent.parent
_LOCAL = Path(__file__).resolve().parent / "config.local.toml"
_local = {}
if tomllib and _LOCAL.exists():
    with open(_LOCAL, "rb") as f:
        _local = tomllib.load(f)


def _get(env, key, default):
    return Path(os.environ.get(env) or _local.get(key) or default).expanduser()


GDELT_DIR = _get("GDELT_DIR", "gdelt_dir", ROOT / "data" / "gdelt")
PLOTS_DIR = _get("PLOTS_DIR", "plots_dir", ROOT / "gdelt_plots")
