"""The import rule of the split: `maskformer` (general) never imports `hgcal` (experiment-specific)."""

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MASKFORMER = REPO / "src" / "maskformer"
MODULES = sorted(MASKFORMER.rglob("*.py"))


def _imported_roots(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(MASKFORMER)))
def test_maskformer_does_not_import_hgcal(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    assert "hgcal" not in set(_imported_roots(tree)), f"{path} imports hgcal (module-level or inside a function)"


def test_the_packages_exist():
    assert MODULES, "src/maskformer has no modules"
    assert (REPO / "src" / "hgcal" / "__init__.py").is_file()
