"""`python -m sv ...`: the `maskformer-sv` command for environments where the package is not installed (PYTHONPATH=src)."""

from sv.scripts.cli import cli

cli(prog_name="maskformer-sv")
