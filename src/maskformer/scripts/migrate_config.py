"""Migration of run configs saved before a package reorganisation (torch-free).

A Lightning run saves its full config (`logs/<run>/config.yaml`) with the `class_path` of every component and the
scale-dict paths as they were at training time. When modules move, those strings go stale. `migrate_config`
rewrites such a file in place: every `class_path` found in the experiment's old -> new map, and every
`scale_dict_path` / `particle_scale_dict_path` still under the legacy `configs/` folder (now `params/`). The
rewrite is line based, so anchors, aliases, comments and number formatting are untouched; the result is re-parsed
and checked to be free of legacy paths (an unknown legacy path raises, so nothing half-migrated is written).

`check_legacy` is what the Lightning CLI runs before parsing: it refuses a config that still names a legacy class
path and tells the user to run the migration command (legacy scale-dict paths are remapped at run time, so they
do not block).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from maskformer.utils.paths import LEGACY_SCALE_DICT_DIR, SCALE_DICT_DIR

SCALE_DICT_KEYS = ("scale_dict_path", "particle_scale_dict_path")
_CLASS_PATH_LINE = re.compile(r"^(?P<head>\s*(?:-\s+)?class_path:\s*)(?P<value>[\w.]+)(?P<rest>.*)$")
_SCALE_DICT_LINE = re.compile(
    r"^(?P<head>\s*(?:-\s+)?(?:particle_)?scale_dict_path:\s*(?:&\w+\s+)?)"
    + re.escape(LEGACY_SCALE_DICT_DIR) + r"/(?P<name>[^\s#]+)(?P<rest>.*)$"
)


def is_legacy_class_path(value: str, class_paths: dict[str, str], prefixes: tuple[str, ...]) -> bool:
    return value in class_paths or (bool(prefixes) and value.startswith(tuple(prefixes)))


def find_legacy(doc, class_paths: dict[str, str], prefixes: tuple[str, ...]) -> list[str]:
    """Every legacy value in a parsed config: class paths in the map or with a legacy prefix, and scale-dict
    paths under the legacy folder. Order of appearance, duplicates kept."""
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "class_path" and isinstance(value, str):
                    if is_legacy_class_path(value, class_paths, prefixes):
                        found.append(value)
                elif key in SCALE_DICT_KEYS and isinstance(value, str):
                    if value.startswith(LEGACY_SCALE_DICT_DIR + "/"):
                        found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    return found


def rewrite_text(text: str, class_paths: dict[str, str]) -> tuple[str, list[tuple[int, str, str]]]:
    """Rewrite the legacy lines of a config text; returns the new text and `(line number, old, new)` per change."""
    lines = text.split("\n")
    changes: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = _CLASS_PATH_LINE.match(line)
        if m and m["value"] in class_paths:
            new = class_paths[m["value"]]
            lines[i] = f"{m['head']}{new}{m['rest']}"
            changes.append((i + 1, m["value"], new))
            continue
        m = _SCALE_DICT_LINE.match(line)
        if m:
            old = f"{LEGACY_SCALE_DICT_DIR}/{m['name']}"
            new = f"{SCALE_DICT_DIR}/{m['name']}"
            lines[i] = f"{m['head']}{new}{m['rest']}"
            changes.append((i + 1, old, new))
    return "\n".join(lines), changes


@dataclass
class Migration:
    path: Path
    changes: list[tuple[int, str, str]] = field(default_factory=list)
    backup_path: Path | None = None
    dry_run: bool = False

    def summary(self) -> str:
        if not self.changes:
            return f"{self.path}: nothing to do"
        head = f"{self.path}: {len(self.changes)} change(s)" + (" (dry run, file untouched)" if self.dry_run else "")
        body = [f"  L{lineno}: {old} -> {new}" for lineno, old, new in self.changes]
        if self.backup_path is not None:
            body.append(f"  original kept as {self.backup_path}")
        return "\n".join([head, *body])


def migrate_config(path, class_paths: dict[str, str], prefixes: tuple[str, ...] = (), *, backup: bool = True,
                   dry_run: bool = False) -> Migration:
    """Rewrite `path` in place (unless `dry_run`), keeping `<path>.bak` when `backup`; idempotent."""
    path = Path(path)
    text = path.read_text()
    new_text, changes = rewrite_text(text, class_paths)
    doc = yaml.safe_load(new_text)
    remaining = sorted(set(find_legacy(doc, class_paths, prefixes))) if doc is not None else []
    if remaining:
        raise ValueError("legacy path(s) with no known replacement: " + ", ".join(remaining)
                         + " (extend the class-path map of the migration)")
    migration = Migration(path, changes, dry_run=dry_run)
    if changes and not dry_run:
        if backup:
            migration.backup_path = path.with_name(path.name + ".bak")
            shutil.copy2(path, migration.backup_path)
        path.write_text(new_text)
    return migration


def config_paths_from_argv(argv) -> list[str]:
    """The config files named by `--config X`, `--config=X` or `-c X` in a Lightning-CLI argument list."""
    paths: list[str] = []
    args = list(argv)
    for i, arg in enumerate(args):
        if arg in ("--config", "-c") and i + 1 < len(args):
            paths.append(args[i + 1])
        elif arg.startswith("--config="):
            paths.append(arg.split("=", 1)[1])
    return paths


def check_legacy(config_paths, class_paths: dict[str, str], prefixes: tuple[str, ...], migrate_command: str) -> None:
    """Refuse (SystemExit) the first config that still names a legacy class path."""
    for path in config_paths:
        p = Path(path)
        if not p.is_file():
            continue  # jsonargparse reports a missing file itself
        try:
            doc = yaml.safe_load(p.read_text())
        except yaml.YAMLError:
            continue
        legacy = [v for v in find_legacy(doc, class_paths, prefixes) if not v.startswith(LEGACY_SCALE_DICT_DIR + "/")]
        if legacy:
            names = ", ".join(sorted(set(legacy)))
            raise SystemExit(
                f"{p} uses class path(s) from before the maskformer/hgcal split: {names}\n"
                f"Rewrite the file in place (a .bak copy is kept) with\n    {migrate_command} {p}\nand rerun the command."
            )
