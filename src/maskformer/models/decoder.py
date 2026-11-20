"""Decoder override for hepattn's dynamic queries."""

from __future__ import annotations

import torch
from hepattn.models.decoder import MaskFormerDecoder
from torch import Tensor


class DynamicQueryDecoder(MaskFormerDecoder):
    """`MaskFormerDecoder` whose dynamic queries work on events with fewer constituents than `num_queries`.

    With `dynamic_queries: true` the queries are the embeddings of the constituents the `query_init` encoder task
    scores above its threshold (at most `num_queries`, the most probable ones; the remaining slots are zero-padded and
    flagged invalid). When NO constituent passes the threshold, hepattn falls back to the `num_queries` most probable
    constituents with `topk(num_queries)`, which raises "k not in range for dimension" on an event with fewer
    constituents than `num_queries` (hepattn 0.2.0). Here the fallback takes at most as many constituents as the event
    has, among the valid ones only; everything else is hepattn's method.
    """

    def initialize_dynamic_queries(self, x: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if not self.dynamic_queries:
            raise ValueError("initialize_dynamic_queries called but decoder.dynamic_queries is False")
        if self.encoder_tasks is None:
            raise ValueError("dynamic_queries=True requires decoder.encoder_tasks to be set")
        query_init_task = next((t for t in self.encoder_tasks if t.name == "query_init"), None)
        if query_init_task is None:
            raise ValueError("dynamic_queries=True requires a task named 'query_init' in encoder_tasks.")
        embed_key, valid_key = f"{self.dynamic_query_source}_embed", f"{self.dynamic_query_source}_valid"
        if embed_key not in x or valid_key not in x:
            raise ValueError(f"dynamic_queries=True requires '{embed_key}' and '{valid_key}' in decoder input dict")

        preds = query_init_task.predict(query_init_task.forward(x))
        prob_key = next((k for k in preds if k.endswith("_prob")), None)
        if prob_key is None:
            raise ValueError(f"query_init task '{query_init_task.name}' did not return a probability prediction.")
        prob = preds[prob_key]  # (1, N)
        if prob.shape[0] != 1:
            raise ValueError(f"dynamic_queries only supports batch_size=1, got {prob.shape[0]}")
        prob, valid = prob[0], x[valid_key][0]

        selected = torch.where((prob >= query_init_task.threshold) & valid)[0]
        if selected.numel() == 0:
            # no seed above threshold: the most probable valid constituents, at most as many as the event has
            k = min(self._num_queries, prob.shape[0])
            selected = prob.masked_fill(~valid, -1.0).topk(k).indices
            selected = selected[valid[selected]]
        elif selected.numel() > self._num_queries:
            selected = selected[prob[selected].topk(self._num_queries).indices]
        selected = selected.sort().values  # the constituents' order
        num_selected = selected.numel()

        embed = x[embed_key][0, selected].detach()
        query_valid = torch.ones(num_selected, dtype=torch.bool, device=embed.device)
        num_padding = self._num_queries - num_selected
        if num_padding > 0:  # zero embeddings, flagged invalid, up to num_queries
            embed = torch.cat([embed, torch.zeros(num_padding, self.dim, device=embed.device, dtype=embed.dtype)], dim=0)
            query_valid = torch.cat([query_valid, torch.zeros(num_padding, dtype=torch.bool, device=embed.device)])
        return embed.unsqueeze(0), query_valid.unsqueeze(0)
