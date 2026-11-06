"""hepattn's Lightning CLI with the repository's scale-dict path resolution and the refusal of legacy configs.

Every `scale_dict_path` of the parsed config (the data module's two dicts and the one of each regression task,
which hepattn opens itself relative to the current directory) is passed through
`maskformer.utils.paths.resolve_scale_dict_path` before anything is instantiated, so that `params/<name>.yaml`
is found from any directory and the `configs/<name>.yaml` of older run configs is mapped onto `params/`. The
resolved strings are what `SaveConfig` writes into the run folder.

An experiment may hand in `legacy_class_paths` (old -> new class-path map) and `legacy_prefixes`: a config file
naming one of those class paths is refused before parsing, with the migration command to run. This matters because
an installed copy of the old packages may still be importable (the container image), in which case the old class
path would silently load stale code instead of failing.
"""

from __future__ import annotations

import sys

from jsonargparse import Namespace
from hepattn.utils.cli import CLI

from maskformer.scripts.migrate_config import check_legacy, config_paths_from_argv
from maskformer.utils.paths import resolve_scale_dict_path

DATA_SCALE_DICT_KEYS = ("data.scale_dict_path", "data.particle_scale_dict_path")
TASKS_KEY = "model.model.init_args.tasks"


class MaskformerCLI(CLI):
    """hepattn `CLI` (run-folder timestamp, best-checkpoint lookup, matmul precision) plus path resolution and
    the legacy-config check."""

    def __init__(
        self,
        *args,
        legacy_class_paths: dict[str, str] | None = None,
        legacy_prefixes: tuple[str, ...] = (),
        migrate_command: str = "maskformer migrate-config",
        **kwargs,
    ):
        # LightningCLI.__init__ parses right away: the legacy settings must be in place before it runs
        self.legacy_class_paths = dict(legacy_class_paths or {})
        self.legacy_prefixes = tuple(legacy_prefixes)
        self.migrate_command = migrate_command
        super().__init__(*args, **kwargs)

    def parse_arguments(self, parser, args) -> None:
        if self.legacy_class_paths and not isinstance(args, (dict, Namespace)):
            argv = sys.argv[1:] if args is None else list(args)
            check_legacy(config_paths_from_argv(argv), self.legacy_class_paths, self.legacy_prefixes, self.migrate_command)
        # LightningCLI.parse_arguments warns whenever `args` is given while sys.argv is not empty, which is always
        # the case under the command line (it builds the argument list itself): parse without the warning.
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)

    def before_instantiate_classes(self) -> None:
        sc = self.config[self.subcommand]
        for key in DATA_SCALE_DICT_KEYS:
            value = sc.get(key)
            if value:
                sc[key] = resolve_scale_dict_path(value)
        for init_args in _task_init_args(sc.get(TASKS_KEY)):
            value = init_args.get("scale_dict_path")
            if value:
                init_args["scale_dict_path"] = resolve_scale_dict_path(value)
        super().before_instantiate_classes()


def _task_init_args(tasks):
    """The `init_args` namespaces of the model's tasks (`tasks: class_path: torch.nn.ModuleList, init_args: modules: [...]`)."""
    modules = getattr(getattr(tasks, "init_args", None), "modules", None) or []
    for task in modules:
        init_args = task.get("init_args") if isinstance(task, Namespace) else None
        if init_args is not None:
            yield init_args
