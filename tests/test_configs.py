"""Static checks of configs/ and params/ (pyyaml only): layout, scale-dict references, task wiring."""

import pathlib
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PARAMS = REPO / "params"
SUBFOLDERS = ("rechit", "layercluster", "trackster_clue3dhigh", "trackster_cluestering", "toycalo", "test", "sv")
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


def _object_dict_key(data: dict) -> str:
    """The data-block key of the object (regression-target) scale dict: `particle_scale_dict_path` for the HGCAL
    configs, `vertex_scale_dict_path` for the SV ones."""
    return "vertex_scale_dict_path" if "vertex_scale_dict_path" in data else "particle_scale_dict_path"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(p.relative_to(CONFIGS)))
def test_scale_dicts_live_in_params(path):
    cfg = yaml.safe_load(path.read_text())
    data = cfg["data"]
    object_key = _object_dict_key(data)
    for key in ("scale_dict_path", object_key):
        value = data[key]
        assert value.startswith("params/"), f"{key}: {value}"
        name = value.split("/", 1)[1]
        assert (PARAMS / name).is_file() or name in KNOWN_MISSING, f"{value} neither in params/ nor known missing"
    tasks = list(_task_scale_dicts(cfg["model"]))
    if path.parent.name == "sv":
        # the SV regression task works in the scaled space and is un-scaled by the prediction writer with the
        # vertex dict of the data block: it carries no dict of its own
        assert not tasks, "the SV regression task takes no scale dict"
    else:
        assert tasks, "no regression task with a scale dict"
        assert all(t == data[object_key] for t in tasks), "the regression task must use the particle dict"
    assert not [s for s in _strings(cfg) if s.startswith("configs/")], "no value may point into configs/"


@pytest.mark.parametrize("path", sorted(PARAMS.glob("*.yaml")), ids=lambda p: p.name)
def test_params_files_are_scale_dicts(path):
    scale_dict = yaml.safe_load(path.read_text())
    assert isinstance(scale_dict, dict) and {"eta", "sinphi", "cosphi"} <= set(scale_dict)
    if path.name.startswith("sv_"):
        assert ("pt" in scale_dict) and ("e" not in scale_dict), "SV dicts are keyed by pT, not energy"
    else:
        assert "e" in scale_dict
    for name, spec in scale_dict.items():
        assert spec["type"] in ("std", "min_max_sym"), f"{path.name}:{name}"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("sv/*.yaml")), ids=lambda p: p.name)
def test_sv_configs_are_consistent(path):
    """The SV configs: inputs/targets known to the feature registry and present in the dicts, sizes in sync."""
    sys.path.insert(0, str(REPO / "src"))
    from sv.dataset.features import TRACK_FEATURES, VERTEX_TARGETS
    from sv.dataset.collections import NUM_CLASSES

    cfg = yaml.safe_load(path.read_text())
    data = cfg["data"]
    inputs, targets = data["inputs"]["track"], data["targets"]["vertex"]
    assert set(inputs) <= set(TRACK_FEATURES) and set(targets) <= set(VERTEX_TARGETS)
    track_dict = yaml.safe_load((REPO / data["scale_dict_path"]).read_text())
    vertex_dict = yaml.safe_load((REPO / data["vertex_scale_dict_path"]).read_text())
    assert set(inputs) <= set(track_dict) and set(targets) <= set(vertex_dict)
    model = cfg["model"]["model"]["init_args"]
    input_net = model["input_nets"]["init_args"]["modules"][0]["init_args"]
    assert input_net["input_name"] == "track" and input_net["net"]["init_args"]["input_size"] == len(inputs)
    assert model["decoder"]["num_queries"] == data["num_objects"]
    tasks = {t["init_args"]["name"]: t for t in model["tasks"]["init_args"]["modules"]}
    assert tasks["regression"]["init_args"]["fields"] == targets
    assert tasks["regression"]["init_args"]["net"]["init_args"]["output_size"] == len(targets)
    assert tasks["mask"]["init_args"]["input_constituent"] == "track" and cfg["model"]["constituent"] == "track"
    assert tasks["classification"]["init_args"]["num_classes"] == cfg["model"]["num_classes"] == NUM_CLASSES == 4
    assert tasks["classification"]["init_args"]["net"]["init_args"]["output_size"] == NUM_CLASSES + 1
    assert len(tasks["classification"]["init_args"]["class_weights"]) == NUM_CLASSES
    # the decoder's cross-attention: mask attention or hepattn's local strided window, with what hepattn asserts
    decoder = model["decoder"]
    assert decoder["mask_attention"] != bool(decoder.get("local_strided_attn", False))
    if decoder.get("local_strided_attn"):
        assert data["batch_size"] == 1 and data["pad_to_max_tracks"] is False and data["sort_tracks_by"] in ("phi", "eta")
        assert decoder["window_size"] % 2 == 0
        assert decoder["decoder_layer_config"].get("attn_kwargs", {}).get("attn_type", "torch") in ("torch", "flex")


def test_params_readme_documents_the_missing_dicts():
    readme = (PARAMS / "README.md").read_text()
    for name in KNOWN_MISSING:
        assert name in readme, f"params/README.md must say that {name} is still to be fitted"


PROJECT_PACKAGES = ("maskformer.", "hgcal.", "sv.")


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
