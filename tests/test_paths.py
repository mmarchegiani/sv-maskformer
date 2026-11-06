"""utils.paths: repository root and scale-dict path resolution (torch-free)."""

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from maskformer.utils.paths import repo_root, resolve_scale_dict_path  # noqa: E402


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """A fake checkout: pyproject.toml, params/x.yaml, HGCAL_REPO_ROOT set and cwd inside it."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "x.yaml").write_text("e: {type: std, mean: 0.0, std: 1.0}\n")
    monkeypatch.setenv("HGCAL_REPO_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_repo_root_from_env(fake_root):
    assert repo_root() == fake_root


def test_repo_root_of_the_checkout(monkeypatch):
    monkeypatch.delenv("HGCAL_REPO_ROOT", raising=False)
    assert repo_root() == REPO


def test_path_as_given_stays_relative(fake_root, capsys):
    assert resolve_scale_dict_path("params/x.yaml") == "params/x.yaml"
    assert capsys.readouterr().out == ""


def test_absolute_path_as_given(fake_root):
    path = fake_root / "params" / "x.yaml"
    assert resolve_scale_dict_path(str(path)) == str(path)


def test_relative_to_the_root_from_another_directory(fake_root, monkeypatch, tmp_path_factory):
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    assert resolve_scale_dict_path("params/x.yaml") == str(fake_root / "params" / "x.yaml")


def test_legacy_configs_path_maps_onto_params(fake_root, capsys):
    assert resolve_scale_dict_path("configs/x.yaml") == "params/x.yaml"
    assert "configs/x.yaml not found, using params/x.yaml" in capsys.readouterr().out


def test_legacy_path_quiet(fake_root, capsys):
    resolve_scale_dict_path("configs/x.yaml", warn=False)
    assert capsys.readouterr().out == ""


def test_legacy_path_from_another_directory(fake_root, monkeypatch, tmp_path_factory):
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    assert resolve_scale_dict_path("configs/x.yaml") == str(fake_root / "params" / "x.yaml")


def test_absolute_path_of_another_machine(fake_root):
    assert resolve_scale_dict_path("/home/other/hgcal-offline-reco/configs/x.yaml") == str(fake_root / "params" / "x.yaml")


def test_missing_lists_the_candidates(fake_root):
    with pytest.raises(FileNotFoundError) as info:
        resolve_scale_dict_path("configs/nope.yaml")
    assert "params/nope.yaml" in str(info.value)
