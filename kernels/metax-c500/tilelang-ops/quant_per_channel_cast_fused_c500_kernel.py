# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""Fused per-channel FP8 quantization kernels.

The kernel quantizes 128-token blocks independently for every hidden
channel.  It optionally gathers/expands tokens and optionally rescales an
FP8 input with per-token, per-128-channel scaling factors before computing
the new per-channel scale.
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from ..kernel_base import Kernel

__all__ = ["PerChannelCastFusedKernel"]

_NUM_PER_TOKENS = 128
_NUM_PER_CHANNELS = 128
_NUM_THREADS = 256
_NUM_THREADS_PER_TOKEN = 64
_PLAIN_THREADS_PER_TOKEN = 16
_SMALL_RESCALE_THREADS_PER_TOKEN = 8
_DEFAULT_RESCALE_THREADS_PER_TOKEN = 16
_LARGE_RESCALE_THREADS_PER_TOKEN = 32
_TILE_K = 64
_C500_SHARED_MEMORY_LIMIT_BYTES = 64 * 1024
_FP8_MAX = 448.0
_MIN_AMAX = 1e-4


def _shared_memory_bytes(
    tile_k: int,
    in_dtype: torch.dtype,
    *,
    register_staging: bool = False,
    threads_per_token: int = _NUM_THREADS_PER_TOKEN,
) -> int:
    """Return explicit staging and reduction shared memory per workgroup."""
    if tile_k <= 0 or threads_per_token <= 0 or tile_k % threads_per_token != 0:
        raise ValueError(
            "tile_k must be positive and divisible by threads_per_token, got "
            f"tile_k={tile_k}, threads_per_token={threads_per_token}"
        )
    element_bytes = torch.empty((), dtype=in_dtype).element_size()
    staging_bytes = 0 if register_staging else _NUM_PER_TOKENS * tile_k * element_bytes
    reduction_bytes = (tile_k // threads_per_token) * _NUM_THREADS * 4
    return staging_bytes + reduction_bytes


def _rescale_threads_per_token(num_tokens_out: int, with_expand: bool) -> int:
    """Choose a C500 FP8 vector width from the static output-token count."""
    if num_tokens_out <= 256:
        return _SMALL_RESCALE_THREADS_PER_TOKEN
    if num_tokens_out >= 4096 or (with_expand and num_tokens_out >= 2048):
        return _LARGE_RESCALE_THREADS_PER_TOKEN
    return _DEFAULT_RESCALE_THREADS_PER_TOKEN


@tilelang.jit(
    out_idx=[1, 2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _per_channel_cast_fused_kernel(
    num_tokens: int,
    num_tokens_out: int,
    hidden: int,
    in_dtype: str,
    with_rescale: bool,
    with_expand: bool,
    round_sf: bool,
    register_staging: bool,
    threads_per_token: int,
):
    tile_m = _NUM_PER_TOKENS
    # Keep the hidden tile at one C500 wave. In particular, a 128x128 FP32
    # staging tile plus reduction scratch needs about 66 KiB of shared memory,
    # exceeding the C500's 64 KiB limit. A fixed tile of 64 is safe for every
    # supported path and also exposes more workgroups than the upstream
    # 128/256-column configurations.
    tile_k = _TILE_K
    vec_k = tile_k // threads_per_token
    vec_m = tile_m * threads_per_token // _NUM_THREADS
    sf_cols = hidden // _NUM_PER_CHANNELS if with_rescale else 1
    pos_size = num_tokens_out if with_expand else 1

    @T.macro
    def _scale_and_inverse(amax: T.float32):
        clamped_amax = T.max(amax, _MIN_AMAX)
        sf = T.alloc_var(T.float32)
        sf_inv = T.alloc_var(T.float32)
        sf = clamped_amax / _FP8_MAX
        sf_inv = _FP8_MAX / clamped_amax
        if round_sf:
            bits = T.reinterpret(sf, T.uint32)
            exp_sf = ((bits - 1) >> 23) + 1 - 127
            sf = T.reinterpret((127 + exp_sf) << 23, T.float32)
            sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
        return sf, sf_inv

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hidden), in_dtype),
        out: T.Tensor((num_tokens_out, hidden), T.float8_e4m3fn),
        out_sf: T.Tensor((T.ceildiv(num_tokens_out, _NUM_PER_TOKENS), hidden), T.float32),
        x_sf_invs: T.Tensor((num_tokens, sf_cols), T.float32),
        pos_to_token: T.Tensor((pos_size,), T.int32),
    ):
        with T.Kernel(
            T.ceildiv(num_tokens_out, tile_m),
            T.ceildiv(hidden, tile_k),
            threads=_NUM_THREADS,
        ) as (pid_token, pid_hidden):
            if register_staging:
                x_staging = T.alloc_local((vec_m, vec_k), in_dtype)
            else:
                x_shared = T.alloc_shared((tile_m, tile_k), in_dtype)
            pos_to_token_local = T.alloc_local((vec_m,), T.int32)
            sf_invs_local = T.alloc_local((vec_m,), T.float32)
            amax_local = T.alloc_local((vec_k,), T.float32)
            amax_shared = T.alloc_shared((vec_k, _NUM_THREADS), T.float32)
            in_local = T.alloc_local((vec_k,), in_dtype)
            out_local = T.alloc_local((vec_k,), T.float8_e4m3fn)

            tid = T.get_thread_binding(0)
            m_id = tid // threads_per_token
            k_id = tid % threads_per_token
            subgroup_lane = tid % 64 // threads_per_token * threads_per_token
            m_offset = pid_token * tile_m + m_id * vec_m
            k_offset = pid_hidden * tile_k + k_id * vec_k
            source_token = T.alloc_var(T.int32)
            logical_value = T.alloc_var(T.float32)

            if with_expand:
                tmp = T.alloc_var(T.int32)
                tmp = -1
                if k_id < vec_m:
                    pos_idx = m_offset + k_id
                    if pos_idx < num_tokens_out:
                        tmp = pos_to_token[pos_idx]
                for i in T.serial(vec_m):
                    pos_to_token_local[i] = T.shfl_sync(tmp, subgroup_lane + i)

            if with_rescale:
                for i in T.serial(vec_m):
                    out_token = m_offset + i
                    source_token = out_token
                    if with_expand:
                        source_token = pos_to_token_local[i]
                    if out_token < num_tokens_out and source_token >= 0:
                        sf_invs_local[i] = x_sf_invs[
                            source_token,
                            (pid_hidden * tile_k + k_id * vec_k) // _NUM_PER_CHANNELS,
                        ]
                    else:
                        sf_invs_local[i] = 0.0

            T.clear(amax_local)
            for i in T.serial(vec_m):
                out_token = m_offset + i
                source_token = out_token
                if with_expand:
                    source_token = pos_to_token_local[i]
                if out_token < num_tokens_out and source_token >= 0:
                    for j in T.vectorized(vec_k):
                        in_local[j] = x[source_token, k_offset + j]
                        if register_staging:
                            x_staging[i, j] = in_local[j]
                        else:
                            x_shared[m_id * vec_m + i, k_id * vec_k + j] = in_local[j]
                    for j in T.vectorized(vec_k):
                        logical_value = T.cast(in_local[j], T.float32)
                        if with_rescale:
                            logical_value = logical_value * sf_invs_local[i]
                        amax_local[j] = T.max(amax_local[j], T.abs(logical_value))
                else:
                    for j in T.vectorized(vec_k):
                        if register_staging:
                            x_staging[i, j] = 0.0
                        else:
                            x_shared[m_id * vec_m + i, k_id * vec_k + j] = 0.0

            for i in T.unroll(vec_k):
                amax_shared[i, tid] = amax_local[i]

            sf = T.alloc_var(T.float32)
            sf_inv = T.alloc_var(T.float32)
            sf = 0.0
            sf_inv = 0.0
            col_id = tid % threads_per_token * vec_k + tid // threads_per_token
            if tid < tile_k:
                for i in T.serial(
                    col_id // vec_k,
                    _NUM_THREADS,
                    threads_per_token,
                ):
                    sf = T.max(sf, amax_shared[col_id % vec_k, i])
                sf, sf_inv = _scale_and_inverse(sf)
                out_sf[pid_token, pid_hidden * tile_k + col_id] = sf
                amax_shared[0, tid] = sf_inv

            for i in T.serial(vec_k):
                amax_local[i] = amax_shared[0, k_id + i * threads_per_token]

            for i in T.serial(vec_m):
                out_token = m_offset + i
                for j in T.vectorized(vec_k):
                    if register_staging:
                        in_local[j] = x_staging[i, j]
                    else:
                        in_local[j] = x_shared[m_id * vec_m + i, k_id * vec_k + j]
                for j in T.vectorized(vec_k):
                    logical_value = T.cast(in_local[j], T.float32)
                    if with_rescale:
                        logical_value = logical_value * sf_invs_local[i]
                    out_local[j] = logical_value * amax_local[j]
                if out_token < num_tokens_out:
                    for j in T.vectorized(vec_k):
                        out[out_token, k_offset + j] = out_local[j]

    return main


class PerChannelCastFusedKernel(Kernel):
    """TileLang implementation shared by the four fused-cast variants."""

    supported_archs: list[int] = [80]

    def __init__(
        self,
        num_tokens: int,
        num_tokens_out: int,
        hidden: int,
        in_dtype: torch.dtype,
        with_rescale: bool,
        with_expand: bool,
        round_sf: bool,
        device: torch.device,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_tokens_out = num_tokens_out
        self.hidden = hidden
        self.dtype = in_dtype
        self.with_rescale = with_rescale
        self.with_expand = with_expand
        self.round_sf = round_sf
        self.device = device
        self.tile_k = _TILE_K
        self.init_config(config, tune)
        self.threads_per_token = self.config["threads_per_token"]
        if self.threads_per_token not in (8, 16, 32, 64):
            raise ValueError(
                "threads_per_token must be one of 8, 16, 32, 64, got "
                f"{self.threads_per_token}"
            )
        self.register_staging = self.config["register_staging"]
        self.shared_memory_bytes = _shared_memory_bytes(
            self.tile_k,
            self.dtype,
            register_staging=self.register_staging,
            threads_per_token=self.threads_per_token,
        )
        if self.shared_memory_bytes > _C500_SHARED_MEMORY_LIMIT_BYTES:
            raise ValueError(
                "per-channel fused cast shared-memory requirement exceeds "
                f"the C500 limit: {self.shared_memory_bytes} > "
                f"{_C500_SHARED_MEMORY_LIMIT_BYTES} bytes"
            )
        self.kernel = _per_channel_cast_fused_kernel(
            num_tokens,
            num_tokens_out,
            hidden,
            self.dtype_str,
            with_rescale,
            with_expand,
            round_sf,
            self.register_staging,
            self.threads_per_token,
        )
        self._dummy_sf = None
        self._dummy_pos = None
        if not with_rescale:
            self._dummy_sf = torch.empty((num_tokens, 1), dtype=torch.float32, device=device)
        if not with_expand:
            self._dummy_pos = torch.empty((1,), dtype=torch.int32, device=device)

    @property
    def default_config(self) -> dict:
        # Input values remain live across the column reduction. Keeping them
        # Thread-local staging removes one shared-memory write/read round trip
        # for the plain paths. Four adjacent hidden values per thread provide
        # real vec4 accesses; a 10-shape C500 ablation beat vec1 and vec2 in
        # every case. Rescale retains shared staging because its FP32 input
        # scales are live at the same time and increase register pressure.
        return {
            "register_staging": not self.with_rescale,
            "threads_per_token": (
                _rescale_threads_per_token(self.num_tokens_out, self.with_expand)
                if self.with_rescale
                else _PLAIN_THREADS_PER_TOKEN
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        x_sf_invs: Optional[torch.Tensor] = None,
        pos_to_token: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sf_arg = x_sf_invs if self.with_rescale else self._dummy_sf
        pos_arg = pos_to_token if self.with_expand else self._dummy_pos
        out, out_sf = self.kernel(x, sf_arg, pos_arg)
        return out, out_sf
