import os
import torch
import tile_kernels


def start_profiler():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()


def stop_profiler():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()


def run_per_channel_plain():
    hidden = int(os.getenv('HIDDEN', '7168'))
    num_tokens = int(os.getenv('NUM_TOKENS', '4096'))
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    def f():
        return tile_kernels.quant.per_channel_cast_fused(x, 'e4m3', num_per_tokens=128, round_sf=False)
    for _ in range(5):
        f()
    start_profiler(); f(); stop_profiler()


def run_per_channel_rescale():
    hidden = int(os.getenv('HIDDEN', '7168'))
    num_tokens = int(os.getenv('NUM_TOKENS', '4096'))
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    xq = tile_kernels.quant.per_token_cast(x, 'e4m3', 128)
    def f():
        return tile_kernels.quant.per_channel_cast_fused(
            xq, 'e4m3', num_per_tokens=128, round_sf=False, num_per_channels=128
        )
    for _ in range(5):
        f()
    start_profiler(); f(); stop_profiler()


def run_batched_transpose():
    hidden = int(os.getenv('HIDDEN', '7168'))
    num_tokens = int(os.getenv('NUM_TOKENS', '8064'))
    experts = int(os.getenv('EXPERTS', '32'))
    dtype_name = os.getenv('DTYPE', 'bf16')
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'e4m3': torch.float8_e4m3fn}[dtype_name]
    x = torch.randn((experts, num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    if dtype == torch.float8_e4m3fn:
        x = x.to(torch.float8_e4m3fn)
    elif dtype == torch.float32:
        x = x.float()
    def f():
        return tile_kernels.transpose.batched_transpose(x)
    for _ in range(5):
        f()
    start_profiler(); f(); stop_profiler()


def run_swiglu_transpose():
    hidden = int(os.getenv('HIDDEN', '7168'))
    num_tokens = int(os.getenv('NUM_TOKENS', '8064'))
    npt = int(os.getenv('NPT', '32'))
    clamp_s = os.getenv('CLAMP', 'none')
    clamp = None if clamp_s == 'none' else float(clamp_s)
    x = torch.randn((num_tokens, hidden * 2), dtype=torch.bfloat16, device='cuda')
    def f():
        return tile_kernels.quant.swiglu_forward_and_per_channel_cast_and_transpose(
            x, 'e4m3', num_per_tokens=npt, round_sf=False,
            without_transpose=False, swiglu_clamp_value=clamp
        )
    for _ in range(5):
        f()
    start_profiler(); f(); stop_profiler()


op = os.environ['PROFILE_OP']
if op == 'pc_plain':
    run_per_channel_plain()
elif op == 'pc_rescale':
    run_per_channel_rescale()
elif op == 'bt':
    run_batched_transpose()
elif op == 'swiglu_t':
    run_swiglu_transpose()
else:
    raise SystemExit(f'unknown PROFILE_OP={op}')
