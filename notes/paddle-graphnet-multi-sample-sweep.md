# PaddlePaddle/GraphNet multi-sample sweep on MetaX C500

Date: 2026-08-13

This note records the stronger follow-up after the one-sample official GraphNet bring-up. The goal was to test whether the generated-code compatibility fix was a one-off hack or a repeatable backend bring-up path across several official PaddleNLP GraphNet samples.

## Experiment

Runner:

- `scripts/official_graphnet_multi_sample_sweep.py`
- `scripts/official_graphnet_static_patch_probe.py`

Raw data:

- `raw/metax-c500-paddle/official_graphnet_multi_sample/`
- `raw/metax-c500-paddle/official_graphnet_multi_sample_v2/`
- `raw/metax-c500-paddle/official_graphnet_multi_sample_final/`

The sweep runs official PaddlePaddle/GraphNet samples through the same compatibility stack:

1. unpack the official GraphNet archive into a temporary workdir
2. apply the minimal backend import compatibility patch
3. run generated-code rewrite from low-level `paddle._C_ops.*` calls to higher-level Paddle APIs
4. execute official `graph_net_bench.paddle.test_compiler --compiler nope`
5. aggregate benchmark status, patch count, e2e median latency, and failure lines

GPU event timing reports `0.0 ms` in this Paddle/MACA image, so e2e wall-clock timing is the reliable metric for this run.

## Compatibility Fix

The first multi-sample sweep passed 3 of 5 samples. Two failures were useful:

| Failure | Root cause | Follow-up |
| --- | --- | --- |
| `ernie-3.0-pico-zh` | invalid sample path for this archive | corrected to `ernie-3.0-tiny-pico-v2-zh` |
| `uer_chinese-roberta-tiny` | generated `_C_ops.full_like` was not covered by the rewrite pass | added `full_like` converter to the static patch runner |

After the fix, the full final sweep passed all five selected official PaddleNLP samples.

## Final Results

Final parameters: 3 warmup iterations, 5 measured trials per sample.

| Official GraphNet Paddle Sample | Generated `_C_ops` Rewritten | Eager e2e Median ms | Compiled/nope e2e Median ms | Status |
| --- | ---: | ---: | ---: | --- |
| `PaddleNLP/ernie-3.0-nano-zh` | 159 | 4.483 | 4.415 | pass |
| `PaddleNLP/ernie-3.0-tiny-pico-v2-zh` | 122 | 3.442 | 3.356 | pass |
| `PaddleNLP/ernie-3.0-tiny-base-v2-zh` | 419 | 12.377 | 12.438 | pass |
| `PaddleNLP/rocketqa-nano-cross-encoder` | 159 | 4.585 | 4.590 | pass |
| `PaddleNLP/uer_chinese-roberta-tiny` | 90 | 3.171 | 3.120 | pass |

Summary:

| Metric | Value |
| --- | ---: |
| Initial sweep success rate | 3 / 5 |
| Final sweep success rate | 5 / 5 |
| Total generated `_C_ops` calls rewritten in final sweep | 949 |
| Fastest final compiled/nope median | 3.120 ms |
| Slowest final compiled/nope median | 12.438 ms |

## Interpretation

This is now stronger than a single demo:

- The project found real GraphNet-on-C500 compatibility gaps rather than stopping at the first failure.
- A concrete generated-code rewrite pass moved the official benchmark path from partial compatibility to 5/5 selected samples.
- The selected samples cover several PaddleNLP graph shapes and generated-code sizes, from 90 to 419 rewritten calls.
- The `compiler=nope` path is expected to be close to eager because it is not an optimizing compiler. The value here is benchmark bring-up and compatibility coverage, not a speedup claim.
- The current Paddle image reports `is_compiled_with_cinn=False`, so the honest next stage is to repeat this with a CINN-enabled image or use vendor profiling to explain static/Predictor overhead.

## Resume-Grade Claim

Safe claim:

> Built a PaddlePaddle/GraphNet compatibility harness on MetaX C500 that rewrites generated low-level `_C_ops` calls to high-level Paddle APIs and runs official `compiler=nope` benchmark timing across five PaddleNLP GraphNet samples; improved selected-sample pass rate from 3/5 to 5/5 and captured e2e median latencies from 3.1 ms to 12.4 ms.

Do not overclaim:

- no CINN speedup was measured
- GPU event timing is not reliable in this image
- this is not yet kernel-level optimization or production inference serving

