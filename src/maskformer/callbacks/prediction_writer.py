"""Base class of the test-time prediction writers.

A MaskFormer test loop yields, per batch, the model outputs, the matched predictions and the losses. This base
class owns everything that does not depend on the experiment's data schema: where the output goes (next to the
checkpoint, `<ckpt>__test[_<suffix>].parquet` and `..._losses.json`), the un-scaling of the regression outputs,
the per-event bookkeeping, the mean test loss per layer/task/term, and writing the parquet. An experiment
implements the hooks:

- `regression_transforms()`: the `{field: transform}` dict used to un-scale the regression outputs;
- `reset_buffers()`: allocate the per-run buffers (called once per test run, after `setup`);
- `collect_batch(inputs, targets, outputs, preds, event_number)`: append one batch to the buffers;
- `build_output(event_numbers)`: the awkward array to write, one entry per event in `event_numbers`.

`plot_chain_readable` tells the evaluation whether the experiment's plot chain can read the written file;
`output_description` only changes the log line.
"""

import json
from pathlib import Path

import awkward as ak
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer


class PredictionWriterBase(Callback):
    #: prefix of the model outputs (the configs' `output_object`)
    output_object = "pflow"
    #: what the log line calls the written entries
    output_description = "predictions"
    #: whether the experiment's plot chain can read the written file
    plot_chain_readable = True
    #: parquet compression of the written file (awkward's default is zstd, which the pyarrow of some LCG views cannot
    #: read; an experiment whose plot chain must run there sets "snappy")
    parquet_compression = "zstd"

    def __init__(self, output_path: str | None = None) -> None:
        super().__init__()
        self._output_path_override = output_path

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "test":
            return
        self.trainer = trainer
        self.ds = trainer.datamodule.test_dataloader().dataset
        self.test_path = getattr(trainer.datamodule, "test_path", None)
        self.test_suff = getattr(trainer.datamodule, "test_suff", None)
        self.var_transform = self.regression_transforms()
        self._event_numbers: list[int] = []
        self._loss_sums: dict[tuple[str, str, str], float] = {}
        self._loss_counts: dict[tuple[str, str, str], int] = {}
        self.reset_buffers()

    # ---- hooks ------------------------------------------------------------------------------

    def regression_transforms(self) -> dict:
        """`{field: transform}` with an `inverse_transform` per regressed field; empty keeps the outputs as they are."""
        return {}

    def reset_buffers(self) -> None:
        """Allocate the per-run buffers filled by `collect_batch`."""

    def collect_batch(self, inputs, targets, outputs, preds, event_number: np.ndarray) -> None:
        """Append one test batch (`event_number`: the events of the batch, in order) to the buffers."""
        raise NotImplementedError

    def build_output(self, event_numbers: np.ndarray) -> ak.Array:
        """The array written to `output_path`, one entry per event of `event_numbers`."""
        raise NotImplementedError

    # ---- shared machinery -------------------------------------------------------------------

    @property
    def output_path(self) -> Path:
        if self._output_path_override is not None:
            return Path(self._output_path_override)
        ckpt_path = Path(self.trainer.ckpt_path)
        suffix = f"_{self.test_suff}" if self.test_suff else ""
        return ckpt_path.parent / f"{ckpt_path.stem}__test{suffix}.parquet"

    @property
    def loss_output_path(self) -> Path:
        return self.output_path.with_name(f"{self.output_path.stem}_losses.json")

    @staticmethod
    def _to_jagged(per_event_arrays: list[np.ndarray]) -> ak.Array:
        """Build a jagged ak.Array from a list of per-event numpy arrays, preserving dtype.

        `ak.Array(per_event_arrays)` builds through a generic iterable path that promotes
        float32 arrays to float64; concatenate+unflatten instead reuses the input buffers.
        """
        counts = np.array([len(arr) for arr in per_event_arrays])
        dtype = per_event_arrays[0].dtype if per_event_arrays else np.float64
        flat = np.concatenate(per_event_arrays) if per_event_arrays else np.array([], dtype=dtype)
        return ak.unflatten(flat, counts)

    def _unscale_regression_preds(self, preds_final: dict) -> dict[str, torch.Tensor]:
        prefix = f"{self.output_object}_"
        raw = {
            key[len(prefix):]: val.detach().cpu()
            for key, val in preds_final["regression"].items()
            if key.startswith(prefix) and "proxy" not in key
        }
        unscaled = {}
        for field, val in raw.items():
            unscaled[field] = self.var_transform[field].inverse_transform(val) if field in self.var_transform else val
        if "sinphi" in unscaled and "cosphi" in unscaled:
            unscaled["phi"] = torch.atan2(unscaled["sinphi"], unscaled["cosphi"])
        return unscaled

    def on_test_batch_end(self, trainer, pl_module, test_step_outputs, batch, batch_idx) -> None:
        inputs, targets = batch
        outputs, preds, losses = test_step_outputs
        event_number = targets["event_number"].cpu().numpy().astype(np.int64)
        self.collect_batch(inputs, targets, outputs, preds, event_number)
        self._event_numbers.extend(int(e) for e in event_number)
        self._accumulate_losses(losses, len(event_number))

    def _accumulate_losses(self, losses: dict, batch_size: int) -> None:
        for layer_name, layer_losses in losses.items():
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    key = (layer_name, task_name, loss_name)
                    self._loss_sums[key] = self._loss_sums.get(key, 0.0) + float(loss_value.detach().cpu()) * batch_size
                    self._loss_counts[key] = self._loss_counts.get(key, 0) + batch_size

    def on_test_end(self, trainer, pl_module) -> None:
        event_numbers = np.array(self._event_numbers, dtype=np.int64)
        out = self.build_output(event_numbers)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ak.to_parquet(out, self.output_path, compression=self.parquet_compression)
        # remembered for whoever runs after the test loop (the plot chain of the evaluation),
        # independent of trainer.ckpt_path surviving teardown
        self.written_output_path = self.output_path
        print(f"Wrote {len(event_numbers)} events with {self.output_description} to {self.output_path}")
        self._write_loss_summary()

    def _write_loss_summary(self) -> None:
        summary: dict = {}
        for (layer_name, task_name, loss_name), total in self._loss_sums.items():
            count = self._loss_counts[(layer_name, task_name, loss_name)]
            summary.setdefault(layer_name, {}).setdefault(task_name, {})[loss_name] = total / count
        summary["total_loss"] = sum(v for layer in summary.values() if isinstance(layer, dict) for task in layer.values() for v in task.values())
        print("Test loss summary:")
        print(json.dumps(summary, indent=2))
        with self.loss_output_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote loss summary to {self.loss_output_path}")
