"""Batched transpose for rank-3 tensors, tuned for MetaX C500.

A block stages one ``[block_x, block_y]`` input tile through shared memory and
stores it back transposed. Each lane reads ``read_vec`` contiguous elements
into registers and scatters them to ``read_vec`` distinct shared rows; the
scatter index is a compile-time constant, so the values never leave registers.
The transpose itself is the swap of the two shared-memory indices.

Keeping that index static is what makes the kernel bandwidth-bound on C500.
The register-tile variant it replaces indexed a thread-local array with a
*runtime* index, which lowers to scratch memory and held the kernel at roughly
half of achievable HBM bandwidth. The tile store stays scalar: widening it
measured strictly worse. See
``docs/summer-camp/batched_transpose_c500_note.md`` for the measurements.
"""

import functools

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["BatchedTransposeKernel", "batched_transpose"]

SUPPORTED_DTYPES = (torch.float8_e4m3fn, torch.bfloat16, torch.float32)

# Tile bytes a single thread should carry. Every dtype measured fastest here:
# fp32 at 16 elements, bf16 at 32, fp8 at 64 all come to 64 bytes, and each is
# within 0.3% of the best config found by a 76-point sweep of block_x, block_y,
# read_vec and threads.
TARGET_BYTES_PER_THREAD = 64

# Past this the C500 memory pipeline falls off a cliff: at 128 bytes per thread
# fp8 measured -40% and bf16 -18% across four shapes. fp32 is unaffected there
# (-0.2%), so this rejects one configuration that would in fact be fine for it;
# the bound is set by the dtypes that do fall off, since a config is built from
# the tile shape without knowing which side of that line it lands on.
MAX_BYTES_PER_THREAD = 64

# Shared memory a single C500 block may allocate. Exceeding it builds cleanly
# and then fails at launch with mcErrorInvalidValue, so it is checked here.
SHARED_MEMORY_BYTES = 64 * 1024

# Width of one shared-memory bank, and the number of banks a request is spread
# over. Used to derive the swizzle below, not to size anything.
SHARED_BANK_BYTES = 4
SHARED_BANKS = 32

# A wavefront is 64 lanes, so one staging step reads ``64 * read_vec *
# elem_bytes`` contiguous bytes. Scalar staging leaves fp8 at 64 B per
# wavefront, half a cache line; widening the *global* read closes that without
# reintroducing a register transpose. Past 8 bytes per lane the gain flattens.
READ_BYTES_PER_LANE = 8

# Contiguous input bytes a tile row should cover. block_y is widened toward
# this before threads are chosen.
TILE_ROW_BYTES = 256

# Below this many 64x64 tiles a workload cannot fill the device, and a halved
# block_x wins: the extra blocks matter more than the shorter contiguous
# store runs. Measured on [8, 192, 256] bf16, 12 tiles per batch — 0.0041 ms
# at block_x=32 against 0.0045 ms at 64.
SMALL_TILE_COUNT = 32

# read_vec for the small-workload branch. These shapes are latency-bound
# rather than bandwidth-bound, so the wider per-lane read wins there even
# though READ_BYTES_PER_LANE is the better target at scale.
SMALL_READ_VEC = 4


def _elem_bytes(dtype: torch.dtype) -> int:
    """Return the size of one ``dtype`` element in bytes."""
    return torch.empty((), dtype=dtype).element_size()


def _read_vec_limit(elem_bytes: int) -> int:
    """Return the widest ``read_vec`` any tile branch may request.

    Callers clamp this by stride alignment. It covers both branches of
    :func:`_default_tile`, so a small workload is not capped at the
    at-scale target.

    Args:
        elem_bytes: Size of one element in bytes.

    Returns:
        A positive element count.
    """
    return max(1, READ_BYTES_PER_LANE // elem_bytes, SMALL_READ_VEC)


def _stride_alignment(row_stride: int, limit: int) -> int:
    """Return the largest power of two up to ``limit`` dividing ``row_stride``.

    A vectorized read of ``read_vec`` elements starts at
    ``row * row_stride + <multiple of read_vec>``, so it is only aligned when
    ``row_stride`` is itself a multiple of ``read_vec``. Callers clamp
    ``read_vec`` to this value rather than rejecting the input.

    Args:
        row_stride: Element stride between consecutive input rows.
        limit: Largest width the caller would otherwise use.

    Returns:
        A power of two in ``[1, limit]``.
    """
    align = 1
    while align < limit and row_stride % (align * 2) == 0:
        align *= 2
    return align


def _default_tile(shape_x: int, shape_y: int, elem_bytes: int,
                  max_read_vec: int) -> dict:
    """Return the tile/thread config measured fastest on C500.

    Workloads too small to fill the device take a halved ``block_x`` for the
    extra blocks. Otherwise ``block_x`` stays at 64, because the
    transposed-from axis is only guaranteed to be a multiple of 64, and
    ``block_y`` widens toward :data:`TILE_ROW_BYTES` of contiguous input for
    every extent that divides evenly. ``read_vec`` targets
    :data:`READ_BYTES_PER_LANE` per lane, clamped by stride alignment.

    Args:
        shape_x: Extent of the axis that is contiguous in the output.
        shape_y: Extent of the axis that is contiguous in the input.
        elem_bytes: Size of one element in bytes.
        max_read_vec: Upper bound on ``read_vec`` from stride alignment.

    Returns:
        A config dict with ``block_x``, ``block_y``, ``read_vec`` and
        ``threads``.
    """
    if (shape_x // 64) * (shape_y // 64) < SMALL_TILE_COUNT:
        return {
            "block_x": 32,
            "block_y": 64,
            "read_vec": min(SMALL_READ_VEC, max_read_vec),
            "threads": 256,
        }
    block_y = 64
    target = max(64, TILE_ROW_BYTES // elem_bytes)
    while (block_y < target
           and shape_y % (block_y * 2) == 0
           and (block_y * 2) * (64 + 1) * elem_bytes <= SHARED_MEMORY_BYTES):
        block_y *= 2
    read_vec = max(1, READ_BYTES_PER_LANE // elem_bytes)
    read_vec = min(read_vec, max_read_vec, block_y)
    # Size the block so each thread carries TARGET_BYTES_PER_THREAD of the
    # tile. The floor keeps a block that cannot reach a full-width tile -- fp8
    # at an extent that will not widen, for instance -- at four wavefronts
    # rather than shrinking it to one.
    threads = max(256, 64 * block_y * elem_bytes // TARGET_BYTES_PER_THREAD)
    return {
        "block_x": 64,
        "block_y": block_y,
        "read_vec": read_vec,
        "threads": threads,
    }


def _swizzle_stride(elem_bytes: int) -> int:
    """Return the XOR-mask step, in elements, between adjacent shared rows.

    Element ``(r, c)`` of the tile is stored at column ``c ^ mask(r)`` with
    ``mask(r) = (r // read_vec) * _swizzle_stride(elem_bytes)``. One step of
    the mask therefore moves a lane exactly one bank along.

    Why a swizzle and not more padding: a staging step has the 32 lanes of a
    conflict group writing 32 *different* shared rows at the same column, so
    their bank indices differ only by the row stride. For a row of ``width``
    elements that stride is ``read_vec * width * elem_bytes /
    SHARED_BANK_BYTES`` banks, and since ``read_vec * elem_bytes`` is pinned to
    ``READ_BYTES_PER_LANE`` (8), it equals ``2 * width`` -- always even, for
    every dtype and every amount of padding. Padding can therefore never spread
    a staging step over more than half the banks; mcProfiler measured 66.67%
    non-conflict access on bf16 [8, 4032, 4096]. XOR-ing the column instead
    makes the 32 lanes land on 32 distinct banks: 100% on the same case.

    Conflict-freedom needs two more things, and both are properties of the tile
    rather than of this layout:

        block_y // read_vec >= 32      32 distinct cols per conflict group
        block_x * elem_bytes >= 128    a shared row spans all 32 banks

    Measured, one instance of each failure mode: fp32 64/64/2 satisfies both and
    reports 100%; bf16 64/64/4 fails the first and reports 66.67%; fp8 64/256/8
    fails the second and reports 70%.

    **They are diagnostic, not a tile-selection rule.** Every conflict-free
    alternative measured slower than the tile the sweep already picked -- bf16
    [8, 4032, 576] 0.99x, fp8 [8, 8064, 4096] 0.92x at 128/256/8/512 and 0.95x
    at 128/128/4, fp8 [8, 8064, 576] 0.97x -- because satisfying them costs
    read width, block count or occupancy on an operator that is already
    DRAM-bound. Some shapes cannot satisfy them at all: fp8 at M = 4032 would
    need block_x = 128, which does not divide 4032. Use these to read a
    profiler report, not to choose a tile.

    Args:
        elem_bytes: Size of one element in bytes.

    Returns:
        Elements per shared-memory bank, at least 1.
    """
    return max(1, SHARED_BANK_BYTES // elem_bytes)


def _store_layout(block_x: int, num_threads: int):
    """Build the fragment layout for the transposed tile store.

    Maps ``(i, j)`` within the ``[block_y, block_x]`` output tile to
    ``(thread, local_index)``. Consecutive ``j`` are contiguous within an
    output row and land on consecutive threads, so the store coalesces.

    Args:
        block_x: Tile extent along the output's contiguous axis.
        num_threads: Threads per block.

    Returns:
        A ``forward_fn`` callable suitable for :class:`T.Fragment`.
    """

    def layout(i, j):
        elems = i * block_x + j
        return elems % num_threads, elems // num_threads

    return layout


def _check_config(block_x: int, block_y: int, read_vec: int, threads: int,
                  elem_bytes: int) -> None:
    """Validate a tile/thread config against what the kernel body can lower.

    Args:
        block_x: Tile extent along the transposed-from axis.
        block_y: Tile extent along the input's contiguous axis.
        read_vec: Elements each lane reads per staging step.
        threads: Threads per block.
        elem_bytes: Size of one element in bytes.

    Raises:
        ValueError: If the tile does not partition evenly across the threads,
            exceeds the measured bytes-per-thread cliff, or stages more
            shared memory than a block may allocate.
    """
    if block_y % read_vec:
        raise ValueError(
            f"block_y ({block_y}) must be a multiple of read_vec ({read_vec})")
    row_threads = block_y // read_vec
    if threads % row_threads:
        raise ValueError(
            f"threads ({threads}) must be a multiple of block_y // read_vec "
            f"({row_threads}): each staging step assigns one lane per "
            "read_vec-wide column group of the tile")
    rows_per_step = threads // row_threads
    if block_x % rows_per_step:
        raise ValueError(
            f"block_x ({block_x}) must be a multiple of threads // "
            f"(block_y // read_vec) ({rows_per_step})")
    bytes_per_thread = block_x * block_y * elem_bytes // threads
    if bytes_per_thread > MAX_BYTES_PER_THREAD:
        raise ValueError(
            f"block_x * block_y * elem_bytes // threads is {bytes_per_thread}, "
            f"above the {MAX_BYTES_PER_THREAD} bytes-per-thread limit; raise "
            "threads or shrink the tile")
    if block_x & (block_x - 1):
        raise ValueError(
            f"block_x ({block_x}) must be a power of two: the shared tile is "
            "swizzled by XOR-ing the column, which only stays inside the row "
            "for a power-of-two width")
    staged = block_y * block_x * elem_bytes
    if staged > SHARED_MEMORY_BYTES:
        raise ValueError(
            f"tile stages {staged} bytes of shared memory, above the "
            f"{SHARED_MEMORY_BYTES} byte per-block limit; shrink block_x or "
            "block_y")


@functools.lru_cache(maxsize=32)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _batched_transpose_kernel(block_x: int, block_y: int, read_vec: int,
                              threads: int, dtype: T.dtype):
    """Build the batched transpose kernel for one tile shape and dtype.

    Args:
        block_x: Tile extent along the transposed-from axis.
        block_y: Tile extent along the input's contiguous axis.
        read_vec: Elements each lane reads per staging step.
        threads: Threads per block.
        dtype: TileLang element dtype.

    Returns:
        The TileLang prim_func for this variant.
    """
    _check_config(block_x, block_y, read_vec, threads, dtype.bytes)
    row_threads = block_y // read_vec
    rows_per_step = threads // row_threads
    steps = block_x // rows_per_step
    swizzle_step = _swizzle_stride(dtype.bytes)
    batches = T.dynamic("num_batches")
    shape_x = T.dynamic("shape_x")
    shape_y = T.dynamic("shape_y")
    stride_x = T.dynamic("stride_x")
    layout = T.Fragment((block_y, block_x), forward_fn=_store_layout(block_x, threads))

    @T.prim_func
    def batched_transpose(
        x: T.StridedTensor[(batches, shape_x, shape_y), (shape_x * stride_x, stride_x, 1), dtype],
        out: T.Tensor[(batches, shape_y, shape_x), dtype],
    ):
        with T.Kernel(shape_y // block_y, shape_x // block_x, batches,
                      threads=threads) as (py, px, pb):
            # The tile is stored swizzled: element (r, c) lives at column
            # c ^ mask(r). That spreads a staging step, whose lanes all target
            # the same column of different rows, over every bank -- which
            # padding cannot do here, see _swizzle_stride(). The phase-3 read
            # stays conflict-free because its lanes share one row, so the mask
            # is a constant permutation of a contiguous block_x-wide range.
            shared = T.alloc_shared((block_y, block_x), dtype)
            tid = T.get_thread_binding()
            row, col = tid // row_threads, tid % row_threads
            swizzle_mask = col * swizzle_step % block_x
            T.assume(shape_x % block_x == 0)
            T.assume(shape_y % block_y == 0)
            T.assume(stride_x % read_vec == 0)
            staged = T.alloc_local((read_vec,), dtype)
            # Load coalesced along the input's contiguous axis, then scatter to
            # read_vec distinct shared rows. The scatter index is static, so it
            # stays in registers -- this is what separates the kernel from the
            # register-transpose form it replaced.
            for step in T.unroll(steps):
                i = step * rows_per_step + row
                for k in T.vectorized(read_vec):
                    staged[k] = x[pb, px * block_x + i, py * block_y + col * read_vec + k]
                # col is r // read_vec for every row r this lane writes, so the
                # mask is one multiply, and the scatter index stays static.
                for k in T.unroll(read_vec):
                    shared[col * read_vec + k, i ^ swizzle_mask] = staged[k]
            T.sync_threads()
            for i, j in T.Parallel(block_y, block_x, loop_layout=layout):
                out[pb, py * block_y + i, px * block_x + j] = \
                    shared[i, j ^ ((i // read_vec) * swizzle_step % block_x)]

    return batched_transpose


def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose the last two dimensions of a rank-3 tensor.

    Args:
        x: A rank-3 CUDA tensor ``[B, M, N]`` with ``M`` and ``N`` multiples
            of 64, unit last stride and a row stride divisible by 4.

    Returns:
        A contiguous ``[B, N, M]`` tensor.

    Raises:
        ValueError: If rank, shape, layout or dtype is unsupported.
    """
    if x.dim() != 3:
        raise ValueError("batched_transpose expects a 3D tensor [B, M, N]")
    batches, shape_x, shape_y = x.shape
    if shape_x % 64 or shape_y % 64:
        raise ValueError("batched_transpose requires M and N divisible by 64")
    if x.stride(-1) != 1 or x.stride(-2) % 4:
        raise ValueError("batched_transpose requires last stride 1 and row stride divisible by 4")
    if x.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unsupported dtype: {x.dtype}")
    out = torch.empty((batches, shape_y, shape_x), dtype=x.dtype, device=x.device)
    if batches and shape_x and shape_y:
        elem_bytes = _elem_bytes(x.dtype)
        max_read_vec = _stride_alignment(x.stride(-2), _read_vec_limit(elem_bytes))
        cfg = _default_tile(shape_x, shape_y, elem_bytes, max_read_vec)
        kernel = _batched_transpose_kernel(
            cfg["block_x"], cfg["block_y"], cfg["read_vec"], cfg["threads"],
            T.dtype(x.dtype))
        kernel(x, out)
    return out


class BatchedTransposeKernel(Kernel):
    """TileLang batched transpose kernel for one static shape/dtype variant.

    Args:
        shape_x: Extent of the axis that is contiguous in the output.
        shape_y: Extent of the axis that is contiguous in the input.
        dtype: Torch element dtype.
        row_stride: Element stride between input rows. Only its power-of-two
            alignment is used, to bound ``read_vec``. Defaults to the
            contiguous stride ``shape_y``.
        config: Optional dict with ``block_x``, ``block_y``, ``read_vec`` and
            ``threads``.
        tune: Accepted for interface parity. This kernel declares no autotune
            search space, so a request falls back to ``config``.

    Raises:
        ValueError: If the resolved tile does not divide the static shape, or
            if a caller-supplied ``read_vec`` exceeds the stride alignment.
    """

    def __init__(self, shape_x: int, shape_y: int, dtype: torch.dtype,
                 row_stride: int | None = None,
                 config: dict | None = None, tune: bool = False):
        super().__init__()
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.dtype = dtype
        self.row_stride = shape_y if row_stride is None else row_stride
        self.init_config(config, tune)
        cfg = self.config
        if shape_x % cfg["block_x"] or shape_y % cfg["block_y"]:
            raise ValueError(
                f"tile ({cfg['block_x']}, {cfg['block_y']}) does not divide "
                f"shape ({shape_x}, {shape_y})")
        if self.row_stride % cfg["read_vec"]:
            raise ValueError(
                f"read_vec ({cfg['read_vec']}) must divide the input row "
                f"stride ({self.row_stride}) for the staging read to be "
                "aligned")
        self.kernel = _batched_transpose_kernel(
            cfg["block_x"], cfg["block_y"], cfg["read_vec"], cfg["threads"],
            T.dtype(dtype))

    @property
    def default_config(self) -> dict:
        """Return the C500-tuned tile config for this kernel's shape."""
        elem_bytes = _elem_bytes(self.dtype)
        max_read_vec = _stride_alignment(self.row_stride, _read_vec_limit(elem_bytes))
        return _default_tile(self.shape_x, self.shape_y, elem_bytes, max_read_vec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose ``x``'s last two dimensions into a fresh contiguous tensor.

        Args:
            x: Input tensor ``[B, shape_x, shape_y]``.

        Returns:
            A contiguous ``[B, shape_y, shape_x]`` tensor.
        """
        out = torch.empty(
            (x.shape[0], self.shape_y, self.shape_x), dtype=x.dtype, device=x.device
        )
        if x.shape[0] and self.shape_x and self.shape_y:
            self.kernel(x, out)
        return out
