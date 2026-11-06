# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch


def concat_mla_absorb_q(q_nope, q_rope):
    can_use_custom_op = (
        q_nope.ndim == q_rope.ndim == 3
        and q_nope.shape[:-1] == q_rope.shape[:-1]
        and (q_nope.shape[-1], q_rope.shape[-1]) == (512, 64)
        and q_nope.dtype == q_rope.dtype == torch.bfloat16
        and q_nope.is_cuda
        and q_rope.is_cuda
        and q_nope.device == q_rope.device
        and q_nope.stride(-1) == q_rope.stride(-1) == 1
        and all(stride % 8 == 0 for stride in q_nope.stride()[:-1])
        and all(stride % 8 == 0 for stride in q_rope.stride()[:-1])
    )
    if not can_use_custom_op:
        return torch.cat([q_nope, q_rope], dim=-1)
    import sgl_kernel

    out = torch.empty(
        (*q_nope.shape[:-1], q_nope.shape[-1] + q_rope.shape[-1]),
        device=q_nope.device,
        dtype=q_nope.dtype,
    )
    torch.ops.sgl_kernel.concat_mla_absorb_q(q_nope, q_rope, out)
    return out
