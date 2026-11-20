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


def test_dynamic_queries_on_events_shorter_than_num_queries():
    """maskformer.models.decoder.DynamicQueryDecoder: an event with fewer tracks than num_queries and no track above
    the seed threshold gets its valid tracks as queries, padded to num_queries (hepattn's fallback raises
    "k not in range for dimension" there); with seeds, the queries are the seeds as in hepattn."""
    from hepattn.models.decoder import MaskFormerDecoder

    from maskformer.models.decoder import DynamicQueryDecoder

    torch.manual_seed(0)
    dim, num_queries = 8, 12
    kwargs = dict(num_queries=num_queries, decoder_layer_config={"dim": dim}, num_decoder_layers=1, dynamic_queries=True, dynamic_query_source="track")
    task = QuerySeedTask(name="query_init", input_object="track", target_field="seed", dim=dim, threshold=0.5)
    decoder, reference = DynamicQueryDecoder(**kwargs), MaskFormerDecoder(**kwargs)
    decoder.encoder_tasks = reference.encoder_tasks = [task]

    def event(n, n_valid=None):
        valid = torch.arange(n) < (n if n_valid is None else n_valid)
        return {"track_embed": torch.randn(1, n, dim), "track_valid": valid.unsqueeze(0)}

    task.threshold = 2.0  # no track can pass: the fallback
    x = event(5)
    with pytest.raises(RuntimeError, match="out of range|not in range"):
        reference.initialize_dynamic_queries(x)
    embed, valid = decoder.initialize_dynamic_queries(x)
    assert embed.shape == (1, num_queries, dim) and valid.shape == (1, num_queries)
    assert valid[0].tolist() == [True] * 5 + [False] * 7
    assert torch.equal(embed[0, :5], x["track_embed"][0]) and (embed[0, 5:] == 0).all()
    embed, valid = decoder.initialize_dynamic_queries(event(1))  # a single track
    assert valid[0].sum() == 1
    embed, valid = decoder.initialize_dynamic_queries(event(20, n_valid=3))  # padding is never a query
    assert valid[0].sum() == 3
    x = event(30)  # a long event: the same queries as hepattn
    for a, b in zip(decoder.initialize_dynamic_queries(x), reference.initialize_dynamic_queries(x)):
        assert torch.equal(a, b)

    for n in (5, 30):  # with seeds: hepattn's selection, on short and long events
        x = event(n)
        prob = next(v for k, v in task.predict(task(x)).items() if k.endswith("_prob"))[0]
        task.threshold = float(prob.median())  # about half of the tracks are seeds (more than num_queries of the 30)
        assert (prob >= task.threshold).any() and (prob < task.threshold).any()
        for a, b in zip(decoder.initialize_dynamic_queries(x), reference.initialize_dynamic_queries(x)):
            assert torch.equal(a, b)
