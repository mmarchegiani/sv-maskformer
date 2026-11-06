"""hepattn's Lightning CLI with the repository's scale-dict path resolution.

Every `scale_dict_path` of the parsed config (the data module's two dicts and the one of each regression task,
which hepattn opens itself relative to the current directory) is passed through `utils.paths.resolve_scale_dict_path`
before anything is instantiated, so that `params/<name>.yaml` is found from any directory and the `configs/<name>.yaml`
of older run configs is mapped onto `params/`. The resolved strings are what `SaveConfig` writes into the run folder.
"""

from __future__ import annotations

from jsonargparse import Namespace
from hepattn.utils.cli import CLI

from utils.paths import resolve_scale_dict_path

DATA_SCALE_DICT_KEYS = ("data.scale_dict_path", "data.particle_scale_dict_path")
TASKS_KEY = "model.model.init_args.tasks"


class HGCCLI(CLI):
    """hepattn `CLI` (run-folder timestamp, best-checkpoint lookup, matmul precision) plus the path resolution."""

    def parse_arguments(self, parser, args) -> None:
        # LightningCLI.parse_arguments warns whenever `args` is given while sys.argv is not empty, which is always
        # the case under the `maskformers` command (it builds the argument list itself): parse without the warning.
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
