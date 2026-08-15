# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""Fused SwiGLU forward + per-channel FP8 cast (+ optional transpose) kernel.

Port of ``swiglu_forward_and_per_channel_cast_and_transpose`` from
MetaX-MACA/TileKernels-Metax (``tile_kernels/quant/``). Fuses the SwiGLU
activation (``silu(gate) * value``) with per-token-block FP8 ``e4m3``
quantization and an optional transpose, emitting a packed FP8 output plus
an fp32 per-block scaling-factor tensor.

The kernel tiles the input as ``TILE_X`` tokens x ``TILE_Y`` hidden
columns; each block reduces the absmax over groups of ``num_per_tokens``
tokens and stores a per-channel (hidden-column) scale.  ``without_transpose``
selects whether the output keeps the ``(num_tokens, hidden)`` layout or is
transposed to ``(hidden, num_tokens)``.

Performance notes (MetaX C500):

- The kernel deliberately does **not** enable ``TL_ENABLE_FAST_MATH``: the
  compiler flag would approximate ``expf``/division and shift the bf16-rounded
  activation by one bf16 quantum at rounding boundaries, breaking the
  bit-exact contract with the torch reference.  The reference TileKernels-Metax
  kernel also runs without fast-math and is compared with ``torch.equal``;
  this kernel must match it.  ``_pack_e4m3`` is bit-exact with ``torch``
  ``fp8_e4m3fn`` casting over the whole ``[-448, 448]`` range including the
  subnormal tail.
- In the ``without_transpose`` path the whole activation tile is
  register-resident: each thread keeps its 4x4 patch(es) in registers from
  the read straight through the packed FP8 output write, so there is no
  ``act_shared`` staging round-trip at all (the original port routed the
  bf16 tile through shared memory twice).  At ``TILE_X=128`` with the
  default 512 threads a thread holds a single 4x4 patch (16 bf16 values =
  8 registers), and even the maximum ``n_patches`` a validated config can
  produce stays far below the register budget, so occupancy is unchanged.
  Consecutive threads still write consecutive hidden columns, so the FP8
  stores coalesce without the shared-memory transpose.
- Both paths read ``x`` as 4x4 register patches (each thread owns
  ``TILE_K`` consecutive rows x ``TILE_K`` consecutive columns), so every
  load instruction touches 1-2 rows instead of the 8 rows a wide
  row-strip read touches.  This measures ~1.3-1.5 TB/s on C500 (vs
  ~0.9-1.0 for the row-strip mapping) with no change to the tile's
  shared-memory footprint, so the fused kernel's occupancy is unchanged.
- The transposed path's ``act_shared`` staging is the only remaining shared
  round-trip.  For ``num_per_tokens=128`` its logical 4x4 transpose write is
  a 4-way bank conflict, so the tile gets two ``TILE_K`` padding columns and
  a swizzled physical layout; measured ~9-13% faster on the three nt=128
  transposed workloads (C500).  ``num_per_tokens=32`` keeps the cheaper
  single-padding unswizzled mapping.
- ``num_threads = 512`` (the port used 256) halves per-thread work in the
  shared-memory phases and improves occupancy.  Measured end-to-end (sum of
  the 10 manifest workloads, MetaX C500): 1.42x over the faithful port.
- The fp32->fp8 write conversion is a hand-written bit pack
  (``_pack_e4m3``), bit-exact with ``torch`` casting, instead of the
  toolchain's software ``__maca_cvt_float2_to_fp8x2`` (the C500 has no
  hardware fp32->fp8 conversion).  This is the dominant write-path cost;
  the pack replaces ~60-ops-per-fp8x2 fp64 emulation with ~12 fp32 bit
  ops.  Isolated write-path speedup on C500: ~1.5x.
- ``num_threads``, ``tile_x``, ``tile_y`` and ``transpose_stages`` are
  config-driven tuning knobs (``config=`` on the Kernel); every config is
  validated against the thread-division and scale-coverage constraints the
  generated loops rely on, so an invalid config raises instead of writing
  garbage.  Default ``transpose_stages`` is 2 for ``num_per_tokens=128``
  and 1 for ``num_per_tokens=32`` (C500 tuning; the port used a fixed 4).
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["QuantSwiGLUChannelCastTransposeKernel"]

# FP8 ``e4m3`` constants (matches ``tile_kernels.quant.common`` for
# ``fmt="e4m3"``): max representable magnitude and the amax floor used to
# keep the scale out of denormal/zero range.
_FP8_MAX_VALUE = 448.0
_SF_CLAMP_MIN = 1e-4


def _validate_tuning_config(
    hidden: int,
    num_per_tokens: int,
    without_transpose: bool,
    in_dtype: str,
    num_threads: int,
    tile_x: int,
    tile_y: int,
    transpose_stages: int,
) -> None:
    """Reject tuning configs that would silently corrupt the output.

    Both paths read ``x`` as 4x4 register patches: each thread covers
    ``TILE_K`` consecutive rows x ``TILE_K`` consecutive columns, so the
    read loops step ``num_threads // (tile_y // TILE_K)`` rows per
    iteration and must divide ``tile_x`` by ``TILE_K`` * that step.  The
    non-transposed path also splits ``num_threads`` across the per-column
    scale reduction; a leftover thread-division compiles but drops output
    columns or leaves shared memory uninitialized.  TILE_K = 4 is fixed in
    the builder.
    """
    if tile_x % num_per_tokens != 0:
        raise ValueError(
            f"tile_x={tile_x} must be divisible by num_per_tokens={num_per_tokens}")
    if num_per_tokens % 4 != 0:
        raise ValueError(
            f"num_per_tokens={num_per_tokens} must be divisible by TILE_K=4 so "
            "a thread's 4-row register patch never straddles a token-block "
            "boundary (the non-transposed FP8 write needs one scale per column)")
    if hidden % tile_y != 0:
        raise ValueError(f"hidden={hidden} must be divisible by tile_y={tile_y}")
    if tile_y % transpose_stages != 0:
        raise ValueError(
            f"tile_y={tile_y} must be divisible by transpose_stages={transpose_stages}")

    num_threads_per_shared_token = tile_y // 4
    if tile_y % 4 != 0:
        raise ValueError(f"tile_y={tile_y} must be divisible by TILE_K=4")
    if num_threads % num_threads_per_shared_token != 0:
        raise ValueError(
            f"num_threads={num_threads} must be divisible by "
            f"num_threads_per_shared_token={num_threads_per_shared_token} "
            "(tile_y // TILE_K)")

    step_shared = num_threads // num_threads_per_shared_token
    if tile_x % (4 * step_shared) != 0:
        raise ValueError(
            f"tile_x={tile_x} must be divisible by TILE_K * "
            f"thread_shared_step = {4 * step_shared} (the read patch covers "
            f"TILE_K=4 rows per thread-group step)")
    if without_transpose:
        num_split = tile_x // num_per_tokens
        if step_shared % num_split != 0:
            raise ValueError(
                f"thread_shared_step={step_shared} must be divisible by "
                f"num_split_blocks={num_split} (tile_x // num_per_tokens)")
        if num_threads < tile_y * num_split:
            raise ValueError(
                f"num_threads={num_threads} must be at least "
                f"tile_y * num_split_blocks = {tile_y * num_split} so every "
                f"scale column is covered by the amax reduction")


@T.macro
def _get_sf_and_inv(amax: float, round_sf: bool):
    """Compute the FP8 scale factor and its inverse from a block absmax.

    Args:
        amax: Absolute max of the quantized block (fp32).
        round_sf: When True, round the scale to the nearest power of two.
    """
    clamped_amax = T.max(amax, _SF_CLAMP_MIN)
    sf = T.alloc_var(T.float32)
    sf = clamped_amax / _FP8_MAX_VALUE
    if not round_sf:
        return sf, _FP8_MAX_VALUE / clamped_amax

    bits = T.reinterpret(sf, T.uint32)
    # amax >= 1e-4 guarantees sign bit = 0 and bits != 0 (no denorm/zero).
    # ``(bits - 1) >> 23 + 1`` gives ceil(log2).
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
    return T.reinterpret((127 + exp_sf) << 23, T.float32), sf_inv


@T.macro
def _pack_e4m3(val: float):
    """Pack an fp32 value into an fp8 e4m3 bit pattern.

    Replaces the toolchain's software ``__maca_cvt_float2_to_fp8x2`` (a
    double-precision, branch-heavy emulation, ~60 ops per fp8x2; the C500 has
    no hardware fp32->fp8 conversion) with a direct fp32 round-to-nearest-even
    to the 3-bit e4m3 mantissa.  The per-block scale guarantees ``|val| <=
    448`` (``sf = amax/448`` with ``|x| <= amax``), so no saturation is needed
    and the toolchain's NaN/overflow branches are unreachable; only the
    denormal path (``abs(val) < 2^-6``) is kept.  Bit-exact with ``torch``
    ``fp8_e4m3fn`` casting over the whole range including the subnormal tail
    (validated element-wise on MetaX C500).

    Args:
        val: The fp32 value to quantize (already scaled by ``sf_inv``).
    """
    bits = T.reinterpret(val, T.uint32)
    sign = ((bits >> 31) & 1) << 7
    ab = bits & 0x7FFFFFFF
    # RNE on the 20 dropped mantissa bits: add half-ulp (0x7FFFF) plus the
    # kept LSB so exact ties round to even; the integer add folds the carry
    # into the exponent field, which is exactly the round-up on mantissa
    # overflow.
    lsb = (ab >> 20) & 1
    r = ab + 0x7FFFF + lsb
    exp8 = ((r >> 23) - 120) & 0xF
    mant = (r >> 20) & 0x7
    normal = sign | (exp8 << 3) | mant
    # Denormals (< 2^-6, min normal) encode in 2^-9 steps: the encoding is the
    # round-to-nearest-even integer of |v| * 512 (0..8).  A rounded 8 carries
    # into the exponent field and encodes the min normal 2^-6 (0x08 / 0x88),
    # matching torch.  (The previous ``reinterpret(round(v*512), u32) & 0x7``
    # masked the IEEE bits of a rounded float -- round(1.0) as fp32 is
    # 0x3F800000, low 3 bits 0 -- so every subnormal encoded as zero; ``& 0x7``
    # also truncated the 8 -> min-normal carry.)  ``T.round`` lowers to C
    # ``roundf`` (half away from zero), so exact halves must be corrected to
    # ties-to-even to match torch: ``|v|*512`` is exactly ``k + 0.5`` iff
    # ``2*w`` is an odd integer; then a rounded-odd result steps one down.
    w = T.reinterpret(ab, T.float32) * 512.0
    m_away = T.cast(T.round(w), T.uint32)
    w2 = w + w
    w2_i = T.cast(w2, T.uint32)
    tie = (w2 == T.cast(w2_i, T.float32)) & ((w2_i & 1) == 1)
    d = m_away - T.if_then_else(tie, m_away & 1, T.uint32(0))
    return T.if_then_else(ab < 0x3C800000, sign | d, normal)


@functools.lru_cache(maxsize=32)
def _quant_swiglu_channel_cast_transpose_kernel(
    hidden: int,
    without_transpose: bool,
    use_clamp: bool,
    round_sf: bool,
    num_per_tokens: int,
    swiglu_clamp_value: Optional[float],
    in_dtype: str,
    num_threads: int,
    tile_x: int,
    tile_y: int,
    transpose_stages: int,
):
    """Build the fused SwiGLU + per-channel FP8 cast kernel."""
    TILE_X, TILE_Y, TILE_K = tile_x, tile_y, 4

    num_threads_per_shared_token = TILE_Y // TILE_K
    thread_shared_step = num_threads // num_threads_per_shared_token
    num_split_blocks = TILE_X // num_per_tokens
    # 4x4 patches each thread owns along the token axis; the register-resident
    # tile (non-transposed path) holds n_patches * TILE_K * TILE_K values.
    n_patches = TILE_X // TILE_K // thread_shared_step

    assert TILE_X % num_per_tokens == 0

    # Runtime symbol: the token axis is dynamic while the hidden axis is a
    # compile-time constant.
    num_tokens = T.dynamic("num_tokens")

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _func():
        @T.prim_func
        def _swiglu_channel_cast_transpose(
            x: T.Tensor[(num_tokens, hidden * 2), in_dtype],
            out: T.Tensor[
                (num_tokens, hidden) if without_transpose else (hidden, num_tokens),
                T.float8_e4m3fn,
            ],
            out_sf: T.Tensor[(num_tokens // num_per_tokens, hidden), T.float32],
        ):
            with T.Kernel(hidden // TILE_Y, num_tokens // TILE_X,
                          threads=num_threads) as (pid_y, pid_x):
                tid = T.get_thread_binding()
                T.assume(num_tokens % TILE_X == 0)

                if without_transpose:
                    # Read x as 4x4 register patches: each thread owns
                    # TILE_K consecutive rows x TILE_K consecutive columns.
                    # Every load instruction then touches only 1-2 rows
                    # instead of the 8 rows a 16-element-wide row-strip reads,
                    # which keeps DRAM page locality high.  The whole tile
                    # stays in registers (no ``act_shared``): with TILE_X=128
                    # and the default 512 threads n_patches=1, so a thread
                    # holds one 4x4 patch (16 bf16 values) from read through
                    # the FP8 write.
                    row, col = tid // num_threads_per_shared_token, tid % num_threads_per_shared_token
                    tmp_l = T.alloc_local((TILE_K,), in_dtype)
                    tmp_r = T.alloc_local((TILE_K,), in_dtype)
                    # Register-resident tile: every thread keeps its
                    # (n_patches x 4x4) patch(es) in registers from read
                    # through the FP8 write, so the non-transposed path
                    # has no shared-memory staging round-trip at all.
                    # Laid out [patch-row, column] so the FP8 store below
                    # vectorizes over the innermost column dim.
                    x_act_reg = T.alloc_local((n_patches * TILE_K, TILE_K), in_dtype)
                    # Per-column absmax accumulated in registers over the
                    # patch's rows while reading, so the absmax needs no
                    # act_shared read-back either.
                    amax_reg = T.alloc_local((TILE_K,), T.float32)
                    for k in T.unroll(TILE_K):
                        amax_reg[k] = 0.0
                    for i_ in T.unroll(n_patches):
                        i = i_ * thread_shared_step + row
                        for j in T.unroll(TILE_K):
                            for k in T.vectorized(TILE_K):
                                tmp_l[k] = x[pid_x * TILE_X + i * TILE_K + j,
                                             pid_y * TILE_Y + col * TILE_K + k]
                            for k in T.vectorized(TILE_K):
                                tmp_r[k] = x[pid_x * TILE_X + i * TILE_K + j,
                                             pid_y * TILE_Y + col * TILE_K + hidden + k]
                            for k in T.unroll(TILE_K):
                                val_l = T.alloc_var(T.float32)
                                val_r = T.alloc_var(T.float32)
                                if use_clamp:
                                    val_l = T.min(tmp_l[k], swiglu_clamp_value)
                                    val_r = T.max(
                                        T.min(tmp_r[k], swiglu_clamp_value),
                                        -swiglu_clamp_value,
                                    )
                                else:
                                    val_l = T.float32(tmp_l[k])
                                    val_r = T.float32(tmp_r[k])
                                val = val_l / (1 + T.exp(-val_l)) * val_r
                                x_act_reg[i_ * TILE_K + j, k] = T.cast(val, in_dtype)
                            # Fold this row's |value| into the register
                            # per-column amax (no act_shared read-back).
                            for k in T.unroll(TILE_K):
                                amax_reg[k] = T.max(
                                    amax_reg[k],
                                    T.abs(T.float32(x_act_reg[i_ * TILE_K + j, k])))

                    amax_shared = T.alloc_shared(
                        (thread_shared_step, TILE_Y), T.bfloat16)
                    sf_shared = T.alloc_shared((num_split_blocks, TILE_Y), T.float32)

                    # Write the register-computed per-column amax partials:
                    # each thread covered its own 4x4 patch's rows, so no
                    # act_shared read-back is needed for the absmax.
                    for j in T.vectorized(TILE_K):
                        amax_shared[row, col * TILE_K + j] = T.cast(amax_reg[j], in_dtype)

                    T.sync_threads()
                    amax_var = T.alloc_var(T.bfloat16, init=0.0)
                    if tid < TILE_Y * num_split_blocks:
                        row_offset = tid // TILE_Y
                        col_offset = tid % TILE_Y
                        for i in T.unroll(thread_shared_step // num_split_blocks):
                            amax_var = T.max(
                                amax_var,
                                amax_shared[row_offset * (thread_shared_step // num_split_blocks) + i, col_offset],
                            )
                        sf, sf_inv = _get_sf_and_inv(T.float32(amax_var), round_sf)
                        out_sf[pid_x * (TILE_X // num_per_tokens) + row_offset, pid_y * TILE_Y + col_offset] = sf
                        sf_shared[row_offset, col_offset] = sf_inv
                    T.sync_threads()

                    # Write the packed FP8 output straight to global
                    # memory from the register tile -- no act_shared
                    # round-trip.  Each thread stores its own 4x4
                    # patch (consecutive threads -> consecutive
                    # columns, so the stores coalesce), scaled by the
                    # per-token-block, per-column inverse scale.
                    # num_per_tokens % 4 == 0 guarantees a thread's
                    # 4-row patch never straddles a token-block
                    # boundary, so one scale holds across the patch's
                    # rows.  The fp32->fp8 conversion uses a
                    # hand-written bit pack (bit-exact with torch,
                    # ~1.4x faster than the toolchain's software
                    # emulation on C500).
                    for i_ in T.unroll(n_patches):
                        i = i_ * thread_shared_step + row
                        blk = (i * TILE_K) // num_per_tokens
                        for j in T.unroll(TILE_K):
                            for k in T.vectorized(TILE_K):
                                v = T.float32(x_act_reg[i_ * TILE_K + j, k]) * sf_shared[blk, col * TILE_K + k]
                                p = _pack_e4m3(v)
                                out[pid_x * TILE_X + i * TILE_K + j, pid_y * TILE_Y + col * TILE_K + k] = (
                                    T.reinterpret(T.cast(p, T.uint8), T.float8_e4m3fn)
                                )

                else:
                    # The transposed output keeps the shared tile: the
                    # (rows x cols) register patches must be exchanged across
                    # threads to land in (cols x rows) order.  nt=128's
                    # logical 4x4 transpose write is a 4-way bank conflict, so
                    # it gets two TILE_K padding columns plus a swizzled
                    # physical layout (measured ~9-13% on C500); nt=32 keeps
                    # the cheaper single-padding unswizzled mapping after
                    # tuning.
                    act_shared = T.alloc_shared(
                        (
                            TILE_Y,
                            TILE_X + TILE_K * (2 if num_per_tokens == 128 else 1),
                        ),
                        in_dtype,
                    )
                    if num_per_tokens == 128:
                        T.annotate_layout(
                            {act_shared: tilelang.layout.make_swizzled_layout(act_shared)}
                        )
                    row, col = tid // num_threads_per_shared_token, tid % num_threads_per_shared_token
                    x_act_local = T.alloc_local((TILE_K, TILE_K), in_dtype)
                    tmp_l = T.alloc_local((TILE_K,), in_dtype)
                    tmp_r = T.alloc_local((TILE_K,), in_dtype)

                    # SwiGLU forward & transpose.
                    for i_ in T.unroll(n_patches):
                        i = i_ * thread_shared_step + row
                        for j in T.unroll(TILE_K):
                            for k in T.vectorized(TILE_K):
                                tmp_l[k] = x[pid_x * TILE_X + i * TILE_K + j, pid_y * TILE_Y + col * TILE_K + k]
                            for k in T.vectorized(TILE_K):
                                tmp_r[k] = x[pid_x * TILE_X + i * TILE_K + j, pid_y * TILE_Y + col * TILE_K + k + hidden]
                            for k in T.unroll(TILE_K):
                                val_l = T.alloc_var(T.float32)
                                val_r = T.alloc_var(T.float32)
                                if use_clamp:
                                    val_l = T.min(tmp_l[k], swiglu_clamp_value)
                                    val_r = T.max(T.min(tmp_r[k], swiglu_clamp_value), -swiglu_clamp_value)
                                else:
                                    val_l = T.float32(tmp_l[k])
                                    val_r = T.float32(tmp_r[k])
                                val = val_l / (1 + T.exp(-val_l)) * val_r
                                x_act_local[k, j] = val

                        for j in T.unroll(TILE_K):
                            for k in T.vectorized(TILE_K):
                                # nt=128's annotated swizzled layout turns
                                # the logical 4-way conflict into a
                                # conflict-free access; nt=32 stays
                                # unswizzled (cheaper after tuning).
                                act_shared[col * TILE_K + j, i * TILE_K + k] = x_act_local[j, k]

                    # Multiple stages reduce register pressure.
                    num_stages = transpose_stages
                    tile_y_per_stage = TILE_Y // num_stages
                    out_fragment = T.alloc_fragment((tile_y_per_stage, TILE_X), T.float32)
                    amax_fragment = T.alloc_fragment(
                        (tile_y_per_stage, TILE_X // num_per_tokens), T.float32)
                    for st in T.unroll(num_stages):
                        T.copy(
                            act_shared[tile_y_per_stage * st: tile_y_per_stage * (st + 1), 0:TILE_X],
                            out_fragment,
                        )
                        out_fragment_reshaped = T.reshape(
                            out_fragment,
                            (tile_y_per_stage, TILE_X // num_per_tokens, num_per_tokens),
                        )
                        T.reduce_absmax(out_fragment_reshaped, amax_fragment, dim=2)

                        for i, j in T.Parallel(tile_y_per_stage, TILE_X // num_per_tokens):
                            sf, sf_inv = _get_sf_and_inv(T.cast(amax_fragment[i, j], T.float32), round_sf)
                            out_sf[pid_x * (TILE_X // num_per_tokens) + j, pid_y * TILE_Y + st * tile_y_per_stage + i] = sf
                            amax_fragment[i, j] = sf_inv

                        for i, j in T.Parallel(tile_y_per_stage, TILE_X):
                            v = out_fragment[i, j] * amax_fragment[i, j // num_per_tokens]
                            p = _pack_e4m3(v)
                            out[pid_y * TILE_Y + st * tile_y_per_stage + i, pid_x * TILE_X + j] = (
                                T.reinterpret(T.cast(p, T.uint8), T.float8_e4m3fn)
                            )

        return _swiglu_channel_cast_transpose

    return _func()


class QuantSwiGLUChannelCastTransposeKernel(Kernel):
    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        hidden: int,
        without_transpose: bool,
        use_clamp: bool,
        round_sf: bool,
        num_per_tokens: int,
        swiglu_clamp_value: Optional[float],
        in_dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.hidden = hidden
        self.without_transpose = without_transpose
        self.use_clamp = use_clamp
        self.round_sf = round_sf
        self.num_per_tokens = num_per_tokens
        self.swiglu_clamp_value = swiglu_clamp_value
        self.dtype = in_dtype
        self.config = config or {}
        tile_x, tile_y = 128, 64
        while tile_x // 2 % num_per_tokens == 0 and hidden % (tile_y * 2) == 0:
            tile_x //= 2
            tile_y *= 2
        tile_x = int(self.config.get("tile_x", tile_x))
        tile_y = int(self.config.get("tile_y", tile_y))
        num_threads = int(self.config.get("num_threads", 512))
        # C500 tuning: the larger nt=128 reduction benefits from two slices,
        # while nt=32 is faster as one copy/reduce/scale/store slice.
        default_transpose_stages = 2 if num_per_tokens == 128 else 1
        transpose_stages = int(
            self.config.get("transpose_stages", default_transpose_stages)
        )
        _validate_tuning_config(
            hidden=hidden,
            num_per_tokens=num_per_tokens,
            without_transpose=without_transpose,
            in_dtype=self.dtype_str,
            num_threads=num_threads,
            tile_x=tile_x,
            tile_y=tile_y,
            transpose_stages=transpose_stages,
        )
        self.kernel = _quant_swiglu_channel_cast_transpose_kernel(
            hidden=hidden,
            without_transpose=without_transpose,
            use_clamp=use_clamp,
            round_sf=round_sf,
            num_per_tokens=num_per_tokens,
            swiglu_clamp_value=swiglu_clamp_value,
            in_dtype=self.dtype_str,
            num_threads=num_threads,
            tile_x=tile_x,
            tile_y=tile_y,
            transpose_stages=transpose_stages,
        )
        self.init_config(config, tune)

    @property
    def dtype_str(self) -> str:
        return str(self.dtype).replace("torch.", "")

    @property
    def default_config(self) -> dict:
        return {}

    def forward(
        self,
        x: torch.Tensor,
        out: torch.Tensor,
        out_sf: torch.Tensor,
    ) -> None:
        """Launch the fused SwiGLU + per-channel FP8 cast kernel.

        Args:
            x: Input (num_tokens, hidden * 2) bf16 tensor.
            out: Output fp8 tensor; (num_tokens, hidden) when
                ``without_transpose`` else (hidden, num_tokens).
            out_sf: Output fp32 scale tensor of shape
                (num_tokens // num_per_tokens, hidden).
        """
        self.kernel(x, out, out_sf)