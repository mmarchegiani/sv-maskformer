"""MaskFormer with a full-event encoder and a decoder split into azimuthal sectors (HEPTv2-style, arXiv:2606.20437).

hepattn is used unmodified. The encoder sees the whole event as in `hepattn.models.MaskFormer`. The decoder runs on the
sectors of each event as separate rows of the decoder batch: `SectoredMaskFormerDecoder` gathers the encoded hits of every
sector (`inputs["sector_node_idx"]`, padded slots flagged False in `inputs["sector_node_valid"]`, both (B, S, M)) into a
batch of B * S rows of M hits and then calls `MaskFormerDecoder.forward` unchanged. Each row carries the decoder's queries
(one shared set of learned initial queries, instantiated per row), so the queries of a sector attend to, update and are
assigned to the hits of that sector only: query -> hit cross-attention, the hit -> query update (bidirectional_ca) and
query self-attention all stay inside the row. The final tasks then run on the same rows (MaskFormer.forward).

Targets come per event as one list of S x T object slots (sector k in slots [kT, (k+1)T), T = decoder queries per sector;
the experiment's sectored dataset, e.g. hgcal.dataset.sectored_dataset). `SectoredMaskFormer.loss` reshapes them to the rows (B * S, T) and calls `MaskFormer.loss`:
one matcher call per step over the stacked layer x row cost matrices, each sector's queries matched to that sector's
targets only, and each loss term hepattn's mean over the rows (not a sum over sectors). With more than one row the mask
task must be maskformer.models.tasks.RowwiseObjectHitMaskTask (hepattn's mask losses mix the pad masks of the rows).

`merge_sector_rows` / `scatter_to_event_hits` turn the row outputs back into one event-level list of S x Q predictions on
the event's hits (the experiment's sectored prediction writer, e.g. hgcal.callbacks.sectored_prediction_writer).
"""

import torch
from hepattn.models import MaskFormer
from hepattn.models.decoder import MaskFormerDecoder
from torch import Tensor

SECTOR_NODE_IDX = "sector_node_idx"
SECTOR_NODE_VALID = "sector_node_valid"


def gather_sector_rows(values: Tensor, sector_node_idx: Tensor, sector_node_valid: Tensor) -> Tensor:
    """Per-hit tensor (B, N, ...) -> the hits of each sector as rows (B * S, M, ...); padded slots are zero (False)."""
    B, S, M = sector_node_idx.shape
    trailing = values.shape[2:]
    index = sector_node_idx.reshape(B, S * M, *([1] * len(trailing))).expand(B, S * M, *trailing)
    rows = torch.gather(values, 1, index).reshape(B * S, M, *trailing)
    keep = sector_node_valid.reshape(B * S, M, *([1] * len(trailing)))
    return torch.where(keep, rows, torch.zeros((), dtype=values.dtype, device=values.device))


def merge_sector_rows(values: Tensor, n_sectors: int) -> Tensor:
    """Row outputs (B * S, Q, ...) -> one event-level list (B, S * Q, ...), sector k in slots [kQ, (k+1)Q)."""
    return values.reshape(values.shape[0] // n_sectors, n_sectors * values.shape[1], *values.shape[2:])


def _event_hit_destinations(sector_node_idx: Tensor, sector_node_valid: Tensor, n_hits: int) -> Tensor:
    # padded slots point at a spare column n_hits that is dropped, so they cannot overwrite a real hit
    return torch.where(sector_node_valid, sector_node_idx, torch.full_like(sector_node_idx, n_hits))


def scatter_to_event_hits(values: Tensor, sector_node_idx: Tensor, sector_node_valid: Tensor, n_hits: int, fill=0) -> Tensor:
    """Row outputs on sector hit slots (B * S, Q, M) -> event-level outputs on the event's hits (B, S * Q, n_hits).

    The queries of sector k carry `fill` on every hit outside sector k."""
    B, S, M = sector_node_idx.shape
    Q = values.shape[1]
    dest = _event_hit_destinations(sector_node_idx, sector_node_valid, n_hits).reshape(B, S, 1, M).expand(B, S, Q, M)
    out = torch.full((B, S, Q, n_hits + 1), fill, dtype=values.dtype, device=values.device)
    out.scatter_(-1, dest, values.reshape(B, S, Q, M))
    return out[..., :n_hits].reshape(B, S * Q, n_hits)


def scatter_rows_to_event_hits(values: Tensor, sector_node_idx: Tensor, sector_node_valid: Tensor, n_hits: int, fill=0) -> Tensor:
    """Per-slot row values (B * S, M) -> per-hit event values (B, n_hits); `fill` on padded hits."""
    B, S, M = sector_node_idx.shape
    dest = _event_hit_destinations(sector_node_idx, sector_node_valid, n_hits).reshape(B, S * M)
    out = torch.full((B, n_hits + 1), fill, dtype=values.dtype, device=values.device)
    out.scatter_(1, dest, values.reshape(B, S * M))
    return out[:, :n_hits]


class SectoredMaskFormerDecoder(MaskFormerDecoder):
    """MaskFormerDecoder run on the sectors of each event as separate rows of the decoder batch (see the module doc).

    Takes MaskFormerDecoder's arguments; `num_queries` is the number of queries PER SECTOR. One input type only, no
    unified decoding, dynamic queries, decoder positional encoding or local strided attention.
    """

    def forward(self, x: dict[str, Tensor], input_names: list[str]) -> tuple[dict[str, Tensor], dict[str, dict]]:
        if len(input_names) != 1 or self.unified_decoding or self.dynamic_queries or self.posenc or self.local_strided_attn:
            raise ValueError(
                "SectoredMaskFormerDecoder supports one input type without unified decoding, dynamic queries, decoder "
                f"posenc or local strided attention (inputs {input_names}, unified_decoding {self.unified_decoding}, "
                f"dynamic_queries {self.dynamic_queries}, posenc {self.posenc}, local_strided_attn {self.local_strided_attn})"
            )
        (name,) = input_names
        inputs = x["inputs"]
        index, valid = inputs[SECTOR_NODE_IDX], inputs[SECTOR_NODE_VALID]
        B, S, M = index.shape
        n_hits = x[f"{name}_valid"].shape[1]
        row_valid = valid.reshape(B * S, M)
        embed = gather_sector_rows(x[f"{name}_embed"], index, valid)
        rows = {
            # the raw per-hit inputs too: the regression task builds its proxy features from them
            "inputs": {
                key: gather_sector_rows(value, index, valid)
                for key, value in inputs.items()
                if key.startswith(f"{name}_") and isinstance(value, Tensor) and value.dim() >= 2 and value.shape[:2] == (B, n_hits)
            },
            f"{name}_embed": embed,
            f"{name}_valid": row_valid,
            "key_embed": embed,
            "key_valid": row_valid,
            f"key_is_{name}": torch.ones_like(row_valid),
        }
        return super().forward(rows, input_names)


def sector_batch_targets(targets: dict[str, Tensor], num_queries: int, target_object: str = "particle") -> dict[str, Tensor]:
    """Event-level targets (B, S * T, ...) -> the decoder rows (B * S, T, ...), with node_valid = the sector hit mask."""
    sector_valid = targets[SECTOR_NODE_VALID]
    B, S, M = sector_valid.shape
    rows = {}
    for key, value in targets.items():
        if key.startswith(target_object + "_") and isinstance(value, Tensor) and value.dim() >= 2 and value.shape[0] == B:
            if value.shape[1] != S * num_queries:
                raise ValueError(
                    f"target '{key}' has {value.shape[1]} object slots per event, expected {S} sectors x {num_queries} queries"
                )
            rows[key] = value.reshape(B * S, num_queries, *value.shape[2:])
        else:
            rows[key] = value
    rows["node_valid"] = sector_valid.reshape(B * S, M)
    if "event_number" in targets:
        rows["event_number"] = targets["event_number"].repeat_interleave(S)
    return rows


class SectoredMaskFormer(MaskFormer):
    """hepattn MaskFormer whose decoder is a SectoredMaskFormerDecoder (see the module doc); takes MaskFormer's arguments.

    `loss` maps the event-level sector targets onto the decoder rows; forward and predict are MaskFormer's, so their
    decoder outputs are per row (B * S, Q, ...).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self.decoder, SectoredMaskFormerDecoder):
            raise TypeError(f"SectoredMaskFormer needs a SectoredMaskFormerDecoder, got {type(self.decoder).__name__}")
        if len(self.input_names) != 1:
            raise ValueError(f"SectoredMaskFormer supports one input type, got {self.input_names}")
        if self.unified_decoding or self.sorter is not None or self.input_sort_field or self.pooling is not None or len(self.encoder_tasks):
            raise ValueError("SectoredMaskFormer supports no unified decoding, sorting, pooling or encoder tasks")

    @property
    def queries_per_sector(self) -> int:
        return self.decoder._num_queries  # noqa: SLF001 - the decoder's own count of learned queries

    def loss(self, outputs: dict, targets: dict) -> tuple[dict, dict, dict]:
        return super().loss(outputs, sector_batch_targets(targets, self.queries_per_sector, self.target_object))
