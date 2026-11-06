"""Overrides of hepattn tasks.

hepattn is used unmodified; behaviour the models here need and hepattn does not offer is added by inheritance.
"""

import torch
from hepattn.models.loss import loss_fns
from hepattn.models.task import IncidenceBasedRegressionTask, ObjectHitMaskTask, ObjectRegressionTask
from hepattn.utils.masks import topk_attn
from torch import Tensor


class RowwiseObjectHitMaskTask(ObjectHitMaskTask):
    """ObjectHitMaskTask whose mask losses are correct for a decoder batch of more than one row.

    hepattn's `mask_bce_loss` and `mask_dice_loss` first select the valid objects of all rows (`pred[object_valid_mask]`,
    shape (n_valid, N)) and then multiply by `input_pad_mask.unsqueeze(1)`, shape (B, 1, N). For B > 1 that pairs every
    object with the pad mask of every row: a mask is normalised by the valid-hit counts of all rows and loses its hits
    beyond the shortest row. B = 1 is unaffected. Here every object's loss uses its own row's pad mask, as hepattn
    computes it for one row, and the loss is the mean over all valid objects of the batch: equal to hepattn for B = 1 and
    the per-object mean for B > 1 (the sectored decoder has one row per sector even at batch size 1).

    mask_bce and mask_dice are computed in one pass over all rows and objects, in float32, with static shapes (no
    selection of the valid objects, no host sync): each object's term is hepattn's formula, and invalid objects enter the
    mean with weight 0. Any other mask loss is hepattn's function evaluated one row at a time. A batch without a valid
    object gives a zero loss (hepattn: NaN, the mean of an empty selection).
    """

    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        layer_outputs: dict[str, dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        # Same inputs as hepattn.models.task.ObjectHitMaskTask.loss; only the reduction over rows differs.
        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_" + self.target_field].type_as(output)

        hit_pad = targets[self.input_constituent + "_valid"]
        object_pad = targets[self.target_object + "_valid"]

        query_mask = targets.get("query_mask")
        if query_mask is not None:
            object_pad = object_pad & query_mask

        sample_weight = target + self.null_weight * (1 - target)
        weight = object_pad.float()
        n_objects = weight.sum().clamp_min(1.0)
        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            if loss_fn in _ROW_FORMULAS:
                per_object = _ROW_FORMULAS[loss_fn](output.float(), target.float(), hit_pad, sample_weight.float())
                losses[loss_fn] = loss_weight * (per_object * weight).sum() / n_objects
            else:
                losses[loss_fn] = loss_weight * self._row_by_row(loss_fn, output, target, object_pad, hit_pad, sample_weight)
        return losses

    @staticmethod
    def _row_by_row(loss_fn, output, target, object_pad, hit_pad, sample_weight) -> Tensor:
        """hepattn's loss function on one row at a time, averaged over all valid objects (losses without a formula here)."""
        n_valid = object_pad.sum(-1).tolist()
        total = None
        for r, n in enumerate(n_valid):
            if n == 0:
                continue
            row = slice(r, r + 1)
            row_loss = loss_fns[loss_fn](
                output[row], target[row], object_valid_mask=object_pad[row], input_pad_mask=hit_pad[row], sample_weight=sample_weight[row]
            )
            total = n * row_loss if total is None else total + n * row_loss
        if total is None:
            # a zero that keeps the graph; padded logits are finfo.min, so their sum would overflow to -inf
            return output.flatten()[:1].sum() * 0.0
        return total / sum(n_valid)


def _mask_bce_per_object(logits: Tensor, target: Tensor, hit_pad: Tensor, sample_weight: Tensor) -> Tensor:
    """hepattn mask_bce_loss per object, (B, Q): the weighted BCE summed over the row's valid hits / their number."""
    pad = hit_pad.unsqueeze(1).to(logits.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, weight=sample_weight, reduction="none") * pad
    return bce.sum(-1) / pad.sum(-1).clamp_min(1.0)


def _mask_dice_per_object(logits: Tensor, target: Tensor, hit_pad: Tensor, sample_weight: Tensor) -> Tensor:
    """hepattn mask_dice_loss per object, (B, Q): 1 - (2 sum(p t) + 1) / (sum p + sum t + 1), p zero on padded hits."""
    probs = logits.sigmoid() * hit_pad.unsqueeze(1).to(logits.dtype)
    numerator = 2 * (probs * target).sum(-1)
    denominator = probs.sum(-1) + target.sum(-1)
    return 1 - (numerator + 1) / (denominator + 1)


_ROW_FORMULAS = {"mask_bce": _mask_bce_per_object, "mask_dice": _mask_dice_per_object}


class ConfigurableIncidenceBasedRegressionTask(IncidenceBasedRegressionTask):
    """IncidenceBasedRegressionTask with a configurable set of "charged" classes.

    hepattn's `get_proxy_feats` takes the proxy features of a query from the track branch whenever its predicted
    class index is < 3 (the PFlow convention: charged hadron / electron / muon). Inputs without tracks (HGCAL,
    for instance: `<input_constituent>_is_track == 0` everywhere) then give every query predicted as a real class
    zero proxy features, and the regression loss pushes the classification head towards the null class (issue #5).

    Args:
        charged_classes: class indices whose proxy features come from the track branch. `None` keeps hepattn's
            default `[0, 1, 2]`; `[]` sends every query through the neutral (energy-weighted) branch, which is the
            setting for track-less inputs. All other arguments are passed to `IncidenceBasedRegressionTask`.
    """

    def __init__(self, *args, charged_classes: list[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.charged_classes = list(range(3)) if charged_classes is None else [int(c) for c in charged_classes]
        self.register_buffer(
            "_charged_class_idx", torch.tensor(self.charged_classes, dtype=torch.long), persistent=False
        )

    def is_charged_query(self, class_probs: Tensor) -> Tensor:
        pred_class = class_probs.argmax(-1)
        if len(self.charged_classes) == 0:
            return torch.zeros_like(pred_class, dtype=torch.bool)
        return torch.isin(pred_class, self._charged_class_idx.to(pred_class.device))

    def get_proxy_feats(
        self,
        incidence: Tensor,
        inputs: dict[str, Tensor],
        class_probs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Identical to hepattn.models.task.IncidenceBasedRegressionTask.get_proxy_feats except for the
        # charged/neutral split, which uses `self.charged_classes` instead of the hard-coded `< 3`.
        proxy_feats = torch.cat(
            [inputs[self.input_constituent + "_" + field].unsqueeze(-1) for field in self.fields],
            dim=-1,
        )

        charged_inc = incidence * inputs[self.input_constituent + "_is_track"].unsqueeze(1)
        charged_inc_top2 = (topk_attn(charged_inc, 2, dim=-2) & (charged_inc > 0)).float()
        charged_inc_max = charged_inc.max(-2, keepdim=True)[0]
        charged_inc_new = (charged_inc == charged_inc_max) & (charged_inc > 0)
        zero_track_mask = charged_inc_new.sum(-1, keepdim=True) == 0
        charged_inc = torch.where(zero_track_mask, charged_inc_top2, charged_inc_new)

        is_charged = self.is_charged_query(class_probs)

        proxy_feats_charged = torch.bmm(charged_inc, proxy_feats)
        proxy_feats_charged[..., 0] = proxy_feats_charged[..., 1] * torch.cosh(proxy_feats_charged[..., 2])
        proxy_feats_charged = self.scale_proxy_feats(proxy_feats_charged) * is_charged.unsqueeze(-1)

        inc_e_weighted = incidence * proxy_feats[..., 0].unsqueeze(1)
        inc_e_weighted *= 1 - inputs[self.input_constituent + "_is_track"].unsqueeze(1)
        inc = inc_e_weighted / (inc_e_weighted.sum(dim=-1, keepdim=True) + 1e-6)

        proxy_feats_neutral = torch.einsum("bnf,bpn->bpf", proxy_feats, inc)
        proxy_feats_neutral[..., 0] = inc_e_weighted.sum(-1)
        proxy_feats_neutral[..., 1] = proxy_feats_neutral[..., 0] / torch.cosh(proxy_feats_neutral[..., 2])

        proxy_feats_neutral = self.scale_proxy_feats(proxy_feats_neutral) * (~is_charged).unsqueeze(-1)
        proxy_feats = proxy_feats_charged + proxy_feats_neutral

        return proxy_feats, is_charged


class ConfigurableObjectRegressionTask(ObjectRegressionTask):
    """ObjectRegressionTask (the regression head reads the query embedding only) with a configurable network and an
    optional contribution to the matching cost.

    hepattn's `ObjectRegressionTask` builds its head as `Dense(dim, len(fields))` with the default hidden layer and
    accepts `cost_weight` without ever computing a cost (the base `Task.cost` returns nothing). Here

    - `net` replaces the head (its `output_size` must equal the number of fields); `None` keeps hepattn's;
    - with `use_cost`, the L1 distance between every query's prediction and every target, averaged over the fields
      and weighted by `cost_weight`, is added to the matching cost (the targets of invalid objects are NaN and are
      ignored by the matcher, so they are replaced by 0 in the cost).

    Used by the SV MaskFormer, whose vertex properties (pT, direction, flight distance) are not aggregates of the
    track inputs, so the incidence-based proxies of the HGCAL regression are not available.
    """

    def __init__(self, *args, net: torch.nn.Module | None = None, use_cost: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if net is not None:
            output_size = getattr(net, "output_size", None)
            if output_size is not None and int(output_size) != self.ndofs:
                raise ValueError(f"net.output_size ({output_size}) must equal the number of regressed fields ({self.ndofs})")
            self.net = net
        self.use_cost = bool(use_cost)

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        if not self.use_cost or self.cost_weight == 0:
            return {}
        pred = outputs[self.regression_key].detach().to(torch.float32)  # (B, Q, F)
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1).to(torch.float32)
        target = torch.nan_to_num(target, nan=0.0)  # (B, T, F); invalid targets are dropped by the matcher
        with torch.autocast(device_type="cuda", enabled=False):
            cost = (pred.unsqueeze(2) - target.unsqueeze(1)).abs().mean(-1)  # (B, Q, T)
        return {f"regression_{self.loss_fn_name}": self.cost_weight * cost}
