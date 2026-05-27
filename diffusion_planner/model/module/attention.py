import torch
import torch.nn as nn


def multihead_attention(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Run nn.MultiheadAttention with odd num_heads support at inference time.

    PyTorch's native fast path (_native_multi_head_attention) requires even num_heads
    when grad is disabled (typical inference). Training keeps grad enabled and uses the
    slow path, which supports odd num_heads (e.g. num_heads=3 from pretrained encoder args).
    """
    if attn.num_heads % 2 == 0:
        return attn(query, key, value, key_padding_mask=key_padding_mask, need_weights=False)[0]

    with torch.enable_grad():
        q = query.detach().requires_grad_(True)
        return attn(q, key, value, key_padding_mask=key_padding_mask, need_weights=False)[0]
