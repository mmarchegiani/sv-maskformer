"""Static checks of configs/ and params/ (pyyaml only): layout, scale-dict references, task wiring."""

import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PARAMS = REPO / "params"
SUBFOLDERS = ("rechit", "layercluster", "trackster_clue3dhigh", "trackster_cluestering", "toycalo", "test")
#: scale dicts referenced by configs but not fitted yet (see params/README.md)
KNOWN_MISSING = {
    "hgcal_mergedsimcluster_var_transform.yaml",
}
CONFIG_FILES = sorted(CONFIGS.glob("*/*.yaml"))


def _strings(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _strings(value)
    elif isinstance(obj, str):
        yield obj


def _class_paths(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "class_path" and isinstance(value, str):
                yield value
            else:
                yield from _class_paths(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _class_paths(value)


def _task_scale_dicts(obj):
    """scale_dict_path of every task that also declares `fields` (the regression tasks)."""
    if isinstance(obj, dict):
        if "scale_dict_path" in obj and "fields" in obj:
            yield obj["scale_dict_path"]
        for value in obj.values():
            yield from _task_scale_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _task_scale_dicts(value)


def test_layout():
    assert not list(CONFIGS.glob("*.yaml")), "configs are grouped by input type, none at the top level"
    for sub in SUBFOLDERS:
        assert (CONFIGS / sub).is_dir(), sub
        assert (CONFIGS / sub / "README.md").is_file(), f"configs/{sub}/README.md"
    assert (CONFIGS / "README.md").is_file() and (PARAMS / "README.md").is_file()
    assert CONFIG_FILES


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_scale_dicts_live_in_params(path):
    cfg = yaml.safe_load(path.read_text())
    data = cfg["data"]
    for key in ("scale_dict_path", "particle_scale_dict_path"):
        value = data[key]
        assert value.startswith("params/"), f"{key}: {value}"
        name = value.split("/", 1)[1]
        assert (PARAMS / name).is_file() or name in KNOWN_MISSING, f"{value} neither in params/ nor known missing"
    tasks = list(_task_scale_dicts(cfg["model"]))
    assert tasks, "no regression task with a scale dict"
    assert all(t == data["particle_scale_dict_path"] for t in tasks), "the regression task must use the particle dict"
    assert not [s for s in _strings(cfg) if s.startswith("configs/")], "no value may point into configs/"


@pytest.mark.parametrize("path", sorted(PARAMS.glob("*.yaml")), ids=lambda p: p.name)
def test_params_files_are_scale_dicts(path):
    scale_dict = yaml.safe_load(path.read_text())
    assert isinstance(scale_dict, dict) and {"e", "eta", "sinphi", "cosphi"} <= set(scale_dict)
    for name, spec in scale_dict.items():
        assert spec["type"] in ("std", "min_max_sym"), f"{path.name}:{name}"


def test_params_readme_documents_the_missing_dicts():
    readme = (PARAMS / "README.md").read_text()
    for name in KNOWN_MISSING:
        assert name in readme, f"params/README.md must say that {name} is still to be fitted"


PROJECT_PACKAGES = ("maskformer.", "hgcal.")


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_class_paths_are_current(path):
    """Every project class path uses the maskformer/hgcal layout, not a pre-split one."""
    sys.path.insert(0, str(REPO / "src"))
    from hgcal.scripts.legacy import LEGACY_CLASS_PATH_PREFIXES, LEGACY_CLASS_PATHS

    cfg = yaml.safe_load(path.read_text())
    project = [c for c in _class_paths(cfg) if not c.startswith(("hepattn.", "torch.", "lightning."))]
    assert project, "no project class path in the config"
    for class_path in project:
        assert class_path.startswith(PROJECT_PACKAGES), class_path
        assert class_path not in LEGACY_CLASS_PATHS and not class_path.startswith(LEGACY_CLASS_PATH_PREFIXES), class_path


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_project_class_paths_import(path):
    """The project classes named by the config exist (container: needs torch and hepattn)."""
    pytest.importorskip("torch")
    pytest.importorskip("hepattn")
    import importlib

    sys.path.insert(0, str(REPO / "src"))
    cfg = yaml.safe_load(path.read_text())
    for class_path in _class_paths(cfg):
        if class_path.startswith(PROJECT_PACKAGES):
            module, _, name = class_path.rpartition(".")
            assert hasattr(importlib.import_module(module), name), class_path
