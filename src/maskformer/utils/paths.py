"""Repo-relative path helpers shared by the datasets, the command line and the torch-free plot chains.

The scale dicts (the normalization parameters written by `maskformers norm-par`) live in `params/` and are
referenced from the training configs as `params/<name>.yaml`, relative to the repository root. Older run
configs (`logs/<run>/config.yaml`, the models on EOS/XRootD) still say `configs/<name>.yaml`, where the files
lived before the 2026-11 refactoring: `resolve_scale_dict_path` maps those onto `params/` so that old runs keep
evaluating without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

#: folder of the scale dicts, relative to the repository root
SCALE_DICT_DIR = "params"
#: where the scale dicts lived before 2026-11 (still found in saved run configs)
LEGACY_SCALE_DICT_DIR = "configs"


def repo_root() -> Path | None:
    """The checkout root: `$HGCAL_REPO_ROOT`, else the first directory above this file, then the current
    directory, then `/app` (where the container image copies the repository) that holds a `pyproject.toml`."""
    env = os.environ.get("HGCAL_REPO_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for candidate in (*here.parents[:6], Path.cwd(), Path("/app")):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def resolve_scale_dict_path(path: str | os.PathLike, warn: bool = True) -> str:
    """Return a path to the scale dict `path` that exists, trying in order:

    1. `path` as given (absolute, or relative to the current directory);
    2. `path` relative to the repository root;
    3. for a legacy `configs/<name>` path, `params/<name>` relative to the current directory, then to the root;
    4. for an absolute path saved on another machine, `<root>/params/<basename>`.

    The first match is returned; a fallback is announced with one line when `warn` is set, so that a saved
    config with an outdated path is still visible in the log. Raises FileNotFoundError listing every candidate.
    A relative result (cases 1 and 3) is kept relative, which keeps the configs saved by Lightning portable.
    """
    given = Path(path)
    if given.is_file():
        return str(path)
    root = repo_root()
    candidates: list[Path] = []
    if not given.is_absolute() and root is not None:
        candidates.append(root / given)
    if given.parts and given.parts[0] == LEGACY_SCALE_DICT_DIR:
        candidates.append(Path(SCALE_DICT_DIR) / given.name)
        if root is not None:
            candidates.append(root / SCALE_DICT_DIR / given.name)
    if given.is_absolute() and root is not None:
        candidates.append(root / SCALE_DICT_DIR / given.name)
    for candidate in candidates:
        if candidate.is_file():
            if warn:
                print(f"[paths] scale dict {path} not found, using {candidate}")
            return str(candidate)
    tried = ", ".join(str(c) for c in [given, *candidates])
    raise FileNotFoundError(f"scale dict {path} not found (tried {tried})")
