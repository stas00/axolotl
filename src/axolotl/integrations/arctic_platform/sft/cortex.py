# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""On-prem Arctic SFT envelope → Cortex Neutrino ``{args, kwargs}``.

Cortex ``fwd_bwd`` is an RPC frame, not the on-prem ``{batch, meta, processing}``
envelope. Short ``loss_fn: sft`` is an on-prem processor name and is rejected
by the zone.

Labels are **pre-rolled** next-token targets (``labels[i] = ids[i+1]``, last
``-100``). Do not also apply an HF causal shift on the server.

``processing`` / ``meta`` are omitted so ``ArcticSFTClient`` cannot inject
``loss_fn: sft`` on the wire.
"""

from __future__ import annotations

from typing import Any

import torch


def _as_long_cpu(t: Any) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    return t.detach().to("cpu", torch.long).contiguous()


def _pad_stack(rows: list[torch.Tensor], pad: int) -> torch.Tensor:
    width = max(int(r.shape[-1]) for r in rows)
    out = []
    for r in rows:
        if r.dim() == 1:
            r = r.unsqueeze(0)
        if r.shape[-1] < width:
            r = torch.nn.functional.pad(r, (0, width - r.shape[-1]), value=pad)
        out.append(r)
    return torch.cat(out, dim=0)


def _flatten_batch(batch: dict) -> dict[str, torch.Tensor]:
    payload = dict(batch)
    inner = payload.get("batch", payload)
    if isinstance(inner, list):
        if not inner:
            raise ValueError("arctic_sft Cortex: empty GAS microbatch list")
        pad = int((payload.get("meta") or {}).get("pad_token_id") or 0)
        keys = ("input_ids", "attention_mask", "labels", "position_ids")
        flat: dict[str, torch.Tensor] = {}
        for key in keys:
            parts = [m[key] for m in inner if key in m]
            if not parts:
                continue
            tensors = [_as_long_cpu(p) for p in parts]
            fill = pad if key != "labels" else -100
            if key == "attention_mask":
                fill = 0
            if key == "position_ids":
                fill = 0
            flat[key] = _pad_stack(tensors, fill)
        return flat
    if not isinstance(inner, dict):
        raise ValueError(
            f"arctic_sft Cortex: expected batch dict or list, got {type(inner)}"
        )
    return {
        k: _as_long_cpu(inner[k])
        for k in inner
        if k in ("input_ids", "attention_mask", "labels", "position_ids")
    }


def _roll_hf_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """HF same-length labels → Neutrino next-token labels (shift left, last -100)."""
    rolled = torch.roll(labels, shifts=-1, dims=-1)
    rolled[..., -1] = -100
    if attention_mask is not None:
        nxt = torch.roll(attention_mask, shifts=-1, dims=-1)
        nxt[..., -1] = 0
        rolled = rolled.masked_fill(nxt == 0, -100)
    return rolled.contiguous()


def to_cortex_sft_payload(batch: dict, *, roll_labels: bool = True) -> dict:
    """Remap an on-prem / Axolotl SFT wire batch to a Cortex ``fwd_bwd`` body."""
    tensors = _flatten_batch(batch)
    input_ids = tensors.get("input_ids")
    if input_ids is None:
        raise ValueError("arctic_sft Cortex: input_ids required")
    attention_mask = tensors.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    labels = tensors.get("labels")
    if labels is None:
        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)
    if roll_labels:
        labels = _roll_hf_labels(input_ids, attention_mask, labels)

    kwargs: dict[str, Any] = {
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "labels": labels.contiguous(),
        "use_cache": False,
    }
    if "position_ids" in tensors:
        kwargs["position_ids"] = tensors["position_ids"].contiguous()
    else:
        kwargs["position_ids"] = (
            torch.arange(input_ids.shape[-1], dtype=torch.long)
            .expand_as(input_ids)
            .contiguous()
        )

    return {
        "args": (),
        "kwargs": kwargs,
        "processing": None,
        "meta": None,
    }
