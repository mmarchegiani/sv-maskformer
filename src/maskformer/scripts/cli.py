"""Click building blocks of a MaskFormer command line.

An experiment package assembles its own `click.Group` from these pieces: the pass-through of Lightning arguments,
the generic `train` and `evaluate` options with their argument-list builders, the `--lightning-help` option and
the `migrate-config` command. Only click is imported here; the entry points are reached through zero-argument
`load_main` callables, so `--help` never imports torch.

    cli = click.Group()
    cli.add_command(make_train_command(lambda: my_experiment.train.main))
    ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

LIGHTNING_SUBCOMMANDS = ("fit", "validate", "test", "predict")
MATMUL_CHOICES = click.Choice(["highest", "high", "medium", "low"])
ATTN_CHOICES = click.Choice(["flash-varlen", "torch"])
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

PASSTHROUGH = dict(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
LIGHTNING_EPILOG = (
    "Any other option is forwarded unchanged to the Lightning CLI, e.g. --trainer.max_epochs 5, "
    "--data.num_workers=0, --model.model.init_args.encoder.init_args.attn_type torch, --print_config. "
    "Forwarded options come last and override the ones above. --lightning-help lists them all."
)


def _opt(flag: str, value) -> list[str]:
    """`[flag, value]` when the option was given, nothing otherwise (the config's value stays in charge)."""
    return [] if value is None else [flag, str(value)]


def _extra(extra_args) -> list[str]:
    """The pass-through arguments, refusing a Lightning subcommand (the command adds its own)."""
    extra = list(extra_args)
    if extra and extra[0] in LIGHTNING_SUBCOMMANDS:
        raise click.UsageError(f"do not pass the Lightning subcommand {extra[0]!r}: the command adds it itself")
    return extra


def _split_features(values) -> list[str]:
    """`--node-features e,eta --node-features cosphi` -> ["e", "eta", "cosphi"]."""
    return [f for value in values for f in value.replace(",", " ").split() if f]


def _apply(decorators, f):
    for decorator in reversed(decorators):  # keep the help output in the listed order
        f = decorator(f)
    return f


def lightning_help_option(subcommand: str, load_main):
    """Eager `--lightning-help`: print the full Lightning CLI help of `subcommand` and exit."""

    def callback(ctx, param, value):
        if not value or ctx.resilient_parsing:
            return
        load_main()([subcommand, "--help"])
        ctx.exit()

    return click.option("--lightning-help", is_flag=True, is_eager=True, expose_value=False, callback=callback,
                        help=f"Print the full Lightning CLI help of `{subcommand}` and exit.")


def extra_args_argument(f):
    return click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)(f)


# ----------------------------------------------------------------------------------------------- train

TRAIN_OPTIONS = [
    click.option("-c", "--config", "configs", multiple=True, required=True, type=click.Path(dir_okay=False),
                 help="Training config. Repeat to layer several configs; later ones override earlier ones."),
    click.option("--name", default=None, help="Run name (--name); the run folder is logs/<name>_<timestamp>."),
    click.option("--ckpt-path", default=None, type=click.Path(dir_okay=False), help="Checkpoint to resume from (--ckpt_path)."),
    click.option("--num-workers", type=int, default=None, help="Data-loader workers (--data.num_workers)."),
    click.option("--max-epochs", type=int, default=None, help="--trainer.max_epochs"),
    click.option("--devices", default=None, help="--trainer.devices"),
    click.option("--precision", default=None, help="--trainer.precision, e.g. bf16-mixed or 32-true."),
    click.option("--matmul-precision", type=MATMUL_CHOICES, default=None, help="float32 matmul precision (--matmul_precision)."),
    click.option("--debug", is_flag=True,
                 help="Single worker and no validation sanity check: --data.num_workers 0 --trainer.num_sanity_val_steps 0."),
]


def train_options(f):
    return _apply(TRAIN_OPTIONS, f)


def train_argv(configs, name, ckpt_path, num_workers, max_epochs, devices, precision, matmul_precision, debug) -> list[str]:
    """The Lightning `fit` argument list of the generic train options."""
    argv = ["fit"]
    for config in configs:
        argv += ["--config", config]
    argv += _opt("--name", name)
    argv += _opt("--ckpt_path", ckpt_path)
    argv += _opt("--data.num_workers", num_workers)
    argv += _opt("--trainer.max_epochs", max_epochs)
    argv += _opt("--trainer.devices", devices)
    argv += _opt("--trainer.precision", precision)
    argv += _opt("--matmul_precision", matmul_precision)
    if debug:
        argv += ["--data.num_workers", "0", "--trainer.num_sanity_val_steps", "0"]
    return argv


def make_train_command(load_main, name: str = "train",
                       help: str = "Train a MaskFormer. The config alone selects the model and the data.") -> click.Command:
    @click.command(name, help=help, epilog=LIGHTNING_EPILOG, **PASSTHROUGH)
    @train_options
    @lightning_help_option("fit", load_main)
    @extra_args_argument
    def train(extra_args, **options):
        load_main()(train_argv(**options) + _extra(extra_args))

    return train


# -------------------------------------------------------------------------------------------- evaluate

EVALUATE_OPTIONS = [
    click.option("-c", "--config", required=True, type=click.Path(dir_okay=False),
                 help="The run's config.yaml (logs/<run>/config.yaml). Without --ckpt-path the best checkpoint "
                      "(lowest val_loss) in <run>/ckpts/ is used."),
    click.option("--ckpt-path", default=None, type=click.Path(dir_okay=False), help="Checkpoint to evaluate (--ckpt_path)."),
    click.option("--test-path", default=None,
                 help="Parquet file or directory to evaluate (--data.test_path); default: the config's test split."),
    click.option("--test-suff", default=None,
                 help="Suffix of the output names, <ckpt>__test_<suff>.parquet (--data.test_suff); "
                      "default: the stem of --test-path."),
    click.option("--num-workers", type=int, default=0, show_default=True, help="Data-loader workers (--data.num_workers)."),
    click.option("--precision", default="32-true", show_default=True, help="--trainer.precision"),
    click.option("--matmul-precision", type=MATMUL_CHOICES, default="highest", show_default=True,
                 help="float32 matmul precision (--matmul_precision)."),
    click.option("--root-dir", default=None, help="Where the outputs of the run go (--trainer.default_root_dir)."),
    click.option("--attn-type", type=ATTN_CHOICES, default=None,
                 help="Encoder attention kernel (--model.model.init_args.encoder.init_args.attn_type); "
                      "`torch` on GPUs without flash attention, e.g. the lxplus T4s."),
    click.option("--skip-plots", is_flag=True, help="Only write the prediction file (--skip_plots true)."),
]


def evaluate_options(f):
    return _apply(EVALUATE_OPTIONS, f)


def evaluate_argv(config, ckpt_path, test_path, test_suff, num_workers, precision, matmul_precision, root_dir,
                  attn_type, skip_plots) -> list[str]:
    """The Lightning `test` argument list of the generic evaluate options."""
    argv = ["test", "--config", config]
    argv += _opt("--ckpt_path", ckpt_path)
    if test_path is not None:
        argv += ["--data.test_path", test_path]
        if test_suff is None:
            test_suff = Path(test_path).stem
    argv += _opt("--data.test_suff", test_suff)
    argv += ["--data.num_workers", str(num_workers)]
    argv += ["--trainer.precision", precision]
    argv += ["--matmul_precision", matmul_precision]
    argv += _opt("--trainer.default_root_dir", root_dir)
    argv += _opt("--model.model.init_args.encoder.init_args.attn_type", attn_type)
    if skip_plots:
        argv += ["--skip_plots", "true"]
    return argv


def make_evaluate_command(load_main, name: str = "evaluate",
                          help: str = "Evaluate a trained MaskFormer: predictions and losses next to the checkpoint.") -> click.Command:
    @click.command(name, help=help, epilog=LIGHTNING_EPILOG, **PASSTHROUGH)
    @evaluate_options
    @lightning_help_option("test", load_main)
    @extra_args_argument
    def evaluate(extra_args, **options):
        load_main()(evaluate_argv(**options) + _extra(extra_args))

    return evaluate


# -------------------------------------------------------------------------------------- migrate-config


def make_migrate_config_command(class_paths: dict[str, str], prefixes: tuple[str, ...] = (),
                                name: str = "migrate-config") -> click.Command:
    """`migrate-config FILE...`: rewrite run configs saved before a package reorganisation (see migrate_config)."""

    @click.command(name)
    @click.argument("files", nargs=-1, required=True, type=click.Path(dir_okay=False, exists=True))
    @click.option("--dry-run", is_flag=True, help="Only report what would change; write nothing.")
    @click.option("--no-backup", is_flag=True, help="Do not keep the original as <file>.bak.")
    def migrate_config(files, dry_run, no_backup):
        """Rewrite the class paths and scale-dict paths of configs saved before the maskformer/hgcal split.

        Each FILE (a run's config.yaml) is rewritten in place and the original kept as FILE.bak. A config that
        still names a legacy class path is refused by `train` and `evaluate` until it is migrated.
        """
        from maskformer.scripts.migrate_config import migrate_config as migrate

        failed = False
        for path in files:
            try:
                result = migrate(path, class_paths, prefixes, backup=not no_backup, dry_run=dry_run)
            except ValueError as exc:
                click.echo(f"{path}: {exc}", err=True)
                failed = True
                continue
            click.echo(result.summary())
        if failed:
            sys.exit(1)

    return migrate_config
