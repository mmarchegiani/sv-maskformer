"""The import rule of the split: `maskformer` (general) never imports an experiment package (`hgcal`, `sv`), and the
experiment packages never import each other (`sv` reuses `maskformer` only)."""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MASKFORMER = REPO / "src" / "maskformer"
SV = REPO / "src" / "sv"
MODULES = sorted(MASKFORMER.rglob("*.py"))
SV_MODULES = sorted(SV.rglob("*.py"))
EXPERIMENTS = ("hgcal", "sv")


def _imported_roots(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(MASKFORMER)))
def test_maskformer_does_not_import_an_experiment(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set(_imported_roots(tree))
    for experiment in EXPERIMENTS:
        assert experiment not in roots, f"{path} imports {experiment} (module-level or inside a function)"


@pytest.mark.parametrize("path", SV_MODULES, ids=lambda p: str(p.relative_to(SV)))
def test_sv_does_not_import_hgcal(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    assert "hgcal" not in set(_imported_roots(tree)), f"{path} imports hgcal"


def test_the_packages_exist():
    assert MODULES, "src/maskformer has no modules"
    assert SV_MODULES, "src/sv has no modules"
    for experiment in EXPERIMENTS:
        assert (REPO / "src" / experiment / "__init__.py").is_file()
