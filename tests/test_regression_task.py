"""maskformer.models.tasks.ConfigurableObjectRegressionTask: the loss of a batch without any valid object is a finite
zero (hepattn's mean over no object is NaN, which happens at batch size 1 for an event without a target), the loss
with targets is hepattn's, and the matching cost is the L1 distance.

    PYTHONPATH=src python -m pytest tests/test_regression_task.py -q
"""

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hepattn")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hepattn.models import Dense  # noqa: E402

from maskformer.models.tasks import ConfigurableObjectRegressionTask, QuerySeedTask  # noqa: E402

FIELDS = ["pt", "eta", "lxy"]


def make_task(dim=8):
    return ConfigurableObjectRegressionTask(name="regression", input_object="query", output_object="pflow", target_object="particle",
                                            fields=FIELDS, loss="l1", loss_weight=2.0, cost_weight=2.0, use_cost=True, dim=dim,
                                            net=Dense(input_size=dim, output_size=len(FIELDS), hidden_layers=[8]))


def batch(task, valid, dim=8, queries=4):
    torch.manual_seed(0)
    x = {"query_embed": torch.randn(1, queries, dim, requires_grad=True)}
    outputs = task(x)
    targets = {"particle_valid": torch.tensor([valid])}
    for field in FIELDS:
        targets[f"particle_{field}"] = torch.where(targets["particle_valid"], torch.randn(1, queries), torch.nan)
    return x, outputs, targets


def test_loss_without_valid_object_is_a_finite_zero():
    task = make_task()
    x, outputs, targets = batch(task, [False] * 4)
    loss = task.loss(outputs, targets)
    assert set(loss) == {"l1"} and torch.isfinite(loss["l1"]) and float(loss["l1"]) == 0.0
    loss["l1"].backward()  # attached to the graph
    assert x["query_embed"].grad is not None


def test_loss_with_targets_and_cost():
    task = make_task()
    x, outputs, targets = batch(task, [True, True, False, False])
    loss = task.loss(outputs, targets)["l1"]
    pred = outputs[task.regression_key][0, :2]
    target = torch.stack([targets[f"particle_{f}"][0, :2] for f in FIELDS], dim=-1)
    assert torch.isfinite(loss) and torch.allclose(loss, 2.0 * (pred - target).abs().mean(-1).mean())
    cost = task.cost(outputs, targets)["regression_l1"]
    assert cost.shape == (1, 4, 4) and torch.allclose(cost[0, :2, :2], 2.0 * (pred.unsqueeze(1) - target.unsqueeze(0)).abs().mean(-1))
    assert torch.isfinite(cost).all()  # the NaN targets of the invalid objects are replaced by 0 in the cost


def test_query_seed_task_loss_is_finite_and_ignores_padding():
    """QuerySeedTask (the `query_init` task of the dynamic queries): finite loss without any seed, padding ignored,
    the hepattn prediction keys the decoder reads."""
    torch.manual_seed(0)
    task = QuerySeedTask(name="query_init", input_object="track", target_field="seed", dim=8, threshold=0.5, loss_weight=3.0)
    x = {"track_embed": torch.randn(1, 6, 8)}
    out = task(x)
    valid = torch.tensor([[True, True, True, True, False, False]])
    no_seed = {"track_seed": torch.zeros(1, 6, dtype=torch.bool), "track_valid": valid}
    assert torch.isfinite(task.loss(out, no_seed)["track_bce"])
    seeds = {"track_seed": torch.tensor([[True, False, False, False, False, False]]), "track_valid": valid}
    loss = task.loss(out, seeds)["track_bce"]
    logit = out["track_logit"][0, :4]
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logit, seeds["track_seed"][0, :4].float(), pos_weight=torch.tensor(3.0))
    assert torch.allclose(loss, 3.0 * expected)
    preds = task.predict(out)
    assert set(preds) == {"track_seed_prob", "track_seed"} and preds["track_seed"].dtype == torch.bool
