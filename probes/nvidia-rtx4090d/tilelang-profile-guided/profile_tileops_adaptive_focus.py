import os
import torch
import tile_kernels
from tile_kernels.testing.generator import generate_topk_idx


def start_profiler():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()


def stop_profiler():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()


def run_pc_moe():
    hidden = int(os.getenv("HIDDEN", "7168"))
    fused_back = os.getenv("FUSED_BACK", "0") in ("1", "true", "True")
    round_sf = os.getenv("ROUND_SF", "0") in ("1", "true", "True")
    params = {
        "num_send_tokens": int(os.getenv("NUM_SEND_TOKENS", "4001")),
        "num_topk": int(os.getenv("NUM_TOPK", "6")),
        "num_experts": int(os.getenv("NUM_EXPERTS", "4")),
        "num_ep_ranks": int(os.getenv("NUM_EP_RANKS", "64")),
        "hidden": hidden,
        "num_per_tokens": 128,
        "num_per_channels": 128,
        "is_fused_cast_back": fused_back,
        "round_sf": round_sf,
    }
    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    _, pos_to_token, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(
        topk_idx, params["num_experts"], 0, 128
    )
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x = tile_kernels.moe.expand_to_fused(x, token_topk_to_pos, pos_to_token)
    if fused_back:
        x = tile_kernels.quant.per_token_cast(x, "e4m3", 128)

    def f():
        return tile_kernels.quant.per_channel_cast_fused(
            x,
            "e4m3",
            num_per_tokens=128,
            round_sf=round_sf,
            num_per_channels=128 if fused_back else None,
            pos_to_token=pos_to_token,
        )

    for _ in range(5):
        f()
    start_profiler()
    f()
    stop_profiler()


def run_swiglu_t32():
    hidden = int(os.getenv("HIDDEN", "2048"))
    num_tokens = int(os.getenv("NUM_TOKENS", "4096"))
    x = torch.randn((num_tokens, hidden * 2), dtype=torch.bfloat16, device="cuda")

    def f():
        return tile_kernels.quant.swiglu_forward_and_per_channel_cast_and_transpose(
            x,
            "e4m3",
            num_per_tokens=32,
            round_sf=os.getenv("ROUND_SF", "1") in ("1", "true", "True"),
            without_transpose=False,
            swiglu_clamp_value=None,
        )

    for _ in range(5):
        f()
    start_profiler()
    f()
    stop_profiler()


op = os.environ["PROFILE_OP"]
if op == "pc_moe":
    run_pc_moe()
elif op == "swiglu_t32":
    run_swiglu_t32()
else:
    raise SystemExit(f"unknown PROFILE_OP={op}")
