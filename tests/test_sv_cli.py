"""The `maskformer-sv` click CLI: help, argument construction, pass-through (no torch; the entry points are replaced
by fakes that record what they were called with).

    PYTHONPATH=src python -m pytest tests/test_sv_cli.py -q
"""

import pathlib
import sys
import types

import pytest

click = pytest.importorskip("click")
from click.testing import CliRunner  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sv.scripts.cli import cli  # noqa: E402

COMMANDS = ("train", "evaluate", "convert", "norm-par", "plot")


@pytest.fixture
def runner():
    return CliRunner()


def fake_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def recording_main(calls, result=None):
    def main(args=None):
        calls.append(list(args))
        return result
    return main


def test_group_help_lists_every_command(runner):
    out = runner.invoke(cli, ["--help"])
    assert out.exit_code == 0, out.output
    for command in COMMANDS:
        assert command in out.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help(runner, command):
    out = runner.invoke(cli, [command, "--help"])
    assert out.exit_code == 0, out.output
    assert "Usage: " in out.output


def test_train_builds_fit_argv(runner, monkeypatch):
    calls = []
    fake_module(monkeypatch, "sv.scripts.train", main=recording_main(calls))
    out = runner.invoke(cli, ["train", "-c", "configs/sv/gv_tracks.yaml", "--max-epochs", "3", "--debug", "--trainer.limit_train_batches", "2"])
    assert out.exit_code == 0, out.output
    assert calls == [["fit", "--config", "configs/sv/gv_tracks.yaml", "--trainer.max_epochs", "3", "--data.num_workers", "0",
                      "--trainer.num_sanity_val_steps", "0", "--trainer.limit_train_batches", "2"]]


def test_evaluate_options(runner, monkeypatch):
    calls = []
    fake_module(monkeypatch, "sv.scripts.evaluate", main=recording_main(calls))
    out = runner.invoke(cli, ["evaluate", "-c", "run/config.yaml", "--test-path", "/data/test", "--plot-dir", "out", "--plot-max-events", "5",
                              "--attn-type", "torch", "--data.num_test", "10"])
    assert out.exit_code == 0, out.output
    (argv,) = calls
    assert argv[:3] == ["test", "--config", "run/config.yaml"]
    pairs = dict(zip(argv[1::2], argv[2::2]))
    assert pairs["--data.test_path"] == "/data/test" and pairs["--data.test_suff"] == "test"
    assert pairs["--plot_dir"] == "out" and pairs["--plot_max_events"] == "5"
    assert pairs["--model.model.init_args.encoder.init_args.attn_type"] == "torch"
    assert argv[-2:] == ["--data.num_test", "10"]


def test_convert_argv(runner, monkeypatch):
    calls = []
    fake_module(monkeypatch, "sv.scripts.dataset.convert_root", main=recording_main(calls, result=0))
    out = runner.invoke(cli, ["convert", "a.root", "b.root", "--outdir", "out", "--split", "0.8", "0.1", "0.1", "--max-events", "7", "--step", "100"])
    assert out.exit_code == 0, out.output
    assert calls == [["a.root", "b.root", "--outdir", "out", "--max-events", "7", "--split", "0.8", "0.1", "0.1", "--step", "100"]]


def test_norm_par_argv_and_exit_code(runner, monkeypatch):
    calls = []
    fake_module(monkeypatch, "sv.scripts.dataset.compute_norm_par", main=recording_main(calls, result=3))
    out = runner.invoke(cli, ["norm-par", "--input", "dir", "--max-events", "5", "--dry-run"])
    assert out.exit_code == 3
    assert calls == [["--input", "dir", "--max-events", "5", "--dry-run"]]


def test_plot_exit_codes(runner, monkeypatch):
    def fake_run_chain(input_path, outdir, **kwargs):
        return {"failures": {"efficiency": "boom"} if "fail" in input_path else {}}

    fake_module(monkeypatch, "sv.scripts.plot.run_sv_plot_chain", run_chain=fake_run_chain)
    assert runner.invoke(cli, ["plot", "--input", "ok.parquet", "--outdir", "o"]).exit_code == 0
    assert runner.invoke(cli, ["plot", "--input", "fail.parquet", "--outdir", "o"]).exit_code == 1
