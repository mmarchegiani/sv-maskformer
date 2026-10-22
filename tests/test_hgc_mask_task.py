"""models.tasks.HGCObjectHitMaskTask: hepattn's mask losses evaluated row by row (decoder batch > 1), hepattn untouched.

The reproducer: a batch of two rows with different numbers of valid hits. The expected mask losses are the mean over the
valid objects of each object's own loss on its own row's hits, computed here without hepattn (`per_object_*`).
hepattn's ObjectHitMaskTask gives a different number for that batch (pad masks broadcast across rows), the same number
for each row alone, and HGCObjectHitMaskTask gives the expected one.
"""

import pytest
import torch
import torch.nn.functional as F

hepattn_task = pytest.importorskip("hepattn.models.task")
from hepattn.models.loss import loss_fns as hepattn_loss_fns  # noqa: E402
from models.tasks import HGCObjectHitMaskTask  # noqa: E402

LOSSES = {"mask_bce": 5.0, "mask_dice": 1.0}
NULL_WEIGHT = 1.0


def make(cls):
    torch.manual_seed(0)
    return cls(
        name="mask", input_constituent="node", input_object="query", output_object="pflow", target_object="particle",
        losses=LOSSES, costs={"mask_dice": 1.0}, dim=8, null_weight=NULL_WEIGHT, logit_scale=4, pred_threshold=0.1,
    )


def make_batch(n_hits=(6, 10), n_objects=(2, 3), num_queries=4, max_nodes=12, seed=1):
    """Row r has n_hits[r] valid hits (the rest padding) and n_objects[r] valid objects first; padded logits are
    finfo.min as ObjectHitMaskTask.forward sets them, padded targets 0."""
    g = torch.Generator().manual_seed(seed)
    B = len(n_hits)
    hit_pad = torch.zeros(B, max_nodes, dtype=torch.bool)
    obj = torch.zeros(B, num_queries, dtype=torch.bool)
    for r in range(B):
        hit_pad[r, : n_hits[r]] = True
        obj[r, : n_objects[r]] = True
    logits = torch.randn(B, num_queries, max_nodes, generator=g) * 2
    logits[~hit_pad.unsqueeze(1).expand_as(logits)] = torch.finfo(logits.dtype).min
    target = (torch.rand(B, num_queries, max_nodes, generator=g) < 0.4) & hit_pad.unsqueeze(1) & obj.unsqueeze(-1)
    outputs = {"pflow_node_logit": logits}
    targets = {"particle_node_valid": target, "node_valid": hit_pad, "particle_valid": obj}
    return outputs, targets


def per_object_bce(logit, target, hit_pad):
    """One object's BCE on its own row: mean over the row's valid hits, hepattn's sample weight."""
    t = target.float()
    w = t + NULL_WEIGHT * (1 - t)
    bce = F.binary_cross_entropy_with_logits(logit, t, weight=w, reduction="none")
    return bce[hit_pad].sum() / hit_pad.sum()


def per_object_dice(logit, target, hit_pad):
    p = logit.sigmoid() * hit_pad
    t = target.float()
    return 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


def expected(outputs, targets):
    logits, target = outputs["pflow_node_logit"], targets["particle_node_valid"]
    hit_pad, obj = targets["node_valid"], targets["particle_valid"]
    bce, dice = [], []
    for r in range(logits.shape[0]):
        for q in torch.nonzero(obj[r]).flatten().tolist():
            bce.append(per_object_bce(logits[r, q], target[r, q], hit_pad[r]))
            dice.append(per_object_dice(logits[r, q], target[r, q], hit_pad[r]))
    return {"mask_bce": LOSSES["mask_bce"] * torch.stack(bce).mean(), "mask_dice": LOSSES["mask_dice"] * torch.stack(dice).mean()}


def row(batch, r):
    outputs, targets = batch
    return {k: v[r : r + 1] for k, v in outputs.items()}, {k: v[r : r + 1] for k, v in targets.items()}


def numbers(losses):
    return {k: round(float(v), 6) for k, v in losses.items()}


def test_reproducer_batch2_unequal_pad_lengths():
    batch = make_batch()
    want = expected(*batch)
    hepattn_loss = make(hepattn_task.ObjectHitMaskTask).loss(*batch)
    ours = make(HGCObjectHitMaskTask).loss(*batch)
    print(f"\nbatch of 2 rows, valid hits (6, 10) of 12, valid objects (2, 3): expected {numbers(want)}, "
          f"hepattn ObjectHitMaskTask {numbers(hepattn_loss)}, HGCObjectHitMaskTask {numbers(ours)}")
    for name in LOSSES:
        torch.testing.assert_close(ours[name], want[name], rtol=1e-5, atol=1e-6)
        # documents hepattn 0.2.0 (the container's): if this fails, hepattn reduces per row itself
        assert not torch.isclose(hepattn_loss[name], want[name], rtol=1e-3), name


@pytest.mark.parametrize("r", [0, 1])
def test_single_row_equals_hepattn(r):
    batch = row(make_batch(), r)
    want = expected(*batch)
    hepattn_loss = make(hepattn_task.ObjectHitMaskTask).loss(*batch)
    ours = make(HGCObjectHitMaskTask).loss(*batch)
    for name in LOSSES:
        torch.testing.assert_close(ours[name], hepattn_loss[name], rtol=1e-6, atol=0)
        torch.testing.assert_close(hepattn_loss[name], want[name], rtol=1e-5, atol=1e-6)


def test_equal_pad_lengths_agree_with_hepattn():
    # with identical pad masks in every row the broadcast is harmless: hepattn and the row-wise loss agree
    batch = make_batch(n_hits=(9, 9, 9), n_objects=(1, 3, 2))
    hepattn_loss = make(hepattn_task.ObjectHitMaskTask).loss(*batch)
    ours = make(HGCObjectHitMaskTask).loss(*batch)
    for name in LOSSES:
        torch.testing.assert_close(ours[name], hepattn_loss[name], rtol=1e-5, atol=1e-6)


def test_rows_without_objects():
    task = make(HGCObjectHitMaskTask)
    batch = make_batch(n_hits=(6, 10, 4), n_objects=(2, 0, 3))
    outputs, targets = batch
    keep = [0, 2]
    sub = ({k: v[keep] for k, v in outputs.items()}, {k: v[keep] for k, v in targets.items()})
    for name, value in task.loss(*batch).items():
        torch.testing.assert_close(value, task.loss(*sub)[name])
    # no valid object anywhere: a finite zero that still carries a graph
    outputs, targets = make_batch(n_hits=(6, 10), n_objects=(0, 0))
    logits = outputs["pflow_node_logit"].clone().requires_grad_(True)
    losses = task.loss({"pflow_node_logit": logits}, targets)
    total = sum(losses.values())
    assert float(total.detach()) == 0.0
    total.backward()
    assert logits.grad is not None


def test_state_dict_unchanged():
    assert set(make(hepattn_task.ObjectHitMaskTask).state_dict()) == set(make(HGCObjectHitMaskTask).state_dict())


@pytest.mark.parametrize("seed", range(6))
def test_vectorised_equals_row_by_row(seed):
    """The one-pass mask_bce / mask_dice equal hepattn's functions evaluated row by row (values and gradients)."""
    g = torch.Generator().manual_seed(100 + seed)
    rows = int(torch.randint(2, 6, (1,), generator=g))
    n_hits = tuple(int(n) for n in torch.randint(1, 13, (rows,), generator=g))
    n_objects = tuple(int(n) for n in torch.randint(0, 5, (rows,), generator=g))
    if sum(n_objects) == 0:
        n_objects = (1,) + n_objects[1:]
    outputs, targets = make_batch(n_hits=n_hits, n_objects=n_objects, num_queries=4, max_nodes=12, seed=seed)
    task = make(HGCObjectHitMaskTask)
    worst = 0.0
    for name, weight in LOSSES.items():
        a = outputs["pflow_node_logit"].clone().requires_grad_(True)
        b = outputs["pflow_node_logit"].clone().requires_grad_(True)
        fast = task.loss({"pflow_node_logit": a}, targets)[name]
        target = targets["particle_node_valid"].type_as(b)
        loop = weight * HGCObjectHitMaskTask._row_by_row(
            name, b, target, targets["particle_valid"], targets["node_valid"], target + NULL_WEIGHT * (1 - target)
        )
        torch.testing.assert_close(fast, loop, rtol=1e-6, atol=1e-7)
        fast.backward()
        loop.backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-8)
        worst = max(worst, float((fast - loop).abs() / loop.abs().clamp_min(1e-12)))
    print(f"\nseed {seed}: rows {rows}, hits {n_hits}, objects {n_objects}: max relative deviation vectorised vs row loop {worst:.2e}")


def test_other_losses_fall_back_to_rows():
    batch = make_batch()
    torch.manual_seed(0)
    task = HGCObjectHitMaskTask(
        name="mask", input_constituent="node", input_object="query", output_object="pflow", target_object="particle",
        losses={"mask_focal": 1.0}, costs={"mask_dice": 1.0}, dim=8, null_weight=NULL_WEIGHT,
    )
    want = sum(n * hepattn_loss_fns["mask_focal"](*[v[r : r + 1] for v in row_args(batch)][:2],
                                                    object_valid_mask=batch[1]["particle_valid"][r : r + 1],
                                                    input_pad_mask=batch[1]["node_valid"][r : r + 1],
                                                    sample_weight=None)
               for r, n in enumerate(batch[1]["particle_valid"].sum(-1).tolist()) if n) / int(batch[1]["particle_valid"].sum())
    torch.testing.assert_close(task.loss(*batch)["mask_focal"], want, rtol=1e-6, atol=1e-7)


def row_args(batch):
    outputs, targets = batch
    return outputs["pflow_node_logit"], targets["particle_node_valid"].float()
