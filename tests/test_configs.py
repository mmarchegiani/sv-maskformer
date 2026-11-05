"""Static checks of configs/ and params/ (pyyaml only): layout, scale-dict references, task wiring."""

import pathlib

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
