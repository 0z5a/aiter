# FlyDSL MoE experiments on gfx1250 (`flydsl_moe_exp`)

Shared branch for studying the FlyDSL grouped-MoE MXFP4 kernels of Yan Dai's DeepSeek-R1 prefill work on gfx1250
(MI450). Since 2026-11-01 it starts at **8ea2265dc**, the head of `dev/yadai_prefill_clean` ("update script and
wheel", 2026-10-29). That is Yan's current line: DSR1 conc6144 gemm1 91 -> 78 us "with boost on", from MMA-group
and DS-read scheduling (`s_wait_alu` removed from the main loop) plus pipeline and prologue tuning.
The kernels are unchanged; this folder adds the measurement tools, and later experiments add commits on top.
Everything needed to reproduce a number in this README is in the branch.

History: until 2026-10-31 this branch sat on `dev/yadai_dsr1_conc6144` @ 2489bae2d (#6984, 2026-10-27) with
`run_dsr1.sh`'s environment. That state is kept as `dev/lixzhang/flydsl_moe_exp_2489bae2d`; the 2026-10-28 results
below were measured there.

## The benchmark

`run_a4w4_conc6144.sh` (repository root): one DeepSeek-R1 MoE layer at prefill, timed kernel by kernel. Since
2026-11-02 the kit's default differs from that script in three ways, to study gemm1 alone:

| | Default since 2026-11-02 | Before (= `run_a4w4_conc6144.sh`) |
|---|---|---|
| Tokens (`MOE_EXP_TOKENS`) | **2048**: 2048 x 8 = 16,384 routed rows, **256 per expert = one full M tile**, no masked rows | 1536: 12,288 routed rows, 192 per expert (rows 192-255 of each tile masked) |
| Timed kernels (`MOE_EXP_KERNELS`) | **gemm1** | quant_a1, gemm1, gemm2 |
| Inputs (`MOE_EXP_ROTATE`, `MOE_EXP_ROTATE_A`) | **4 rotating copies** of W1 + scales: every launch reads the experts' weights from HBM (`MOE_EXP_ROTATE_A=1` rotates A + scales too) | one copy, reused by every launch |

The test still runs the whole MoE once and checks it against its torch reference; untimed kernels print
`us = nan` and a "timing unavailable" warning. `MOE_EXP_TOKENS=1536 MOE_EXP_KERNELS=all MOE_EXP_ROTATE=1` is the
earlier benchmark.

| | |
|---|---|
| Shape | 64 experts, 2048 tokens (default), top-8 routing, model_dim 7168, inter_dim 2048 |
| Routing | balanced (`AITER_MOE_EXPERT_BALANCE=true`): token *t* goes to experts 8*t* .. 8*t*+7 (mod 64), so every expert gets tokens x 8 / 64 rows |
| Data | MXFP4 activations and weights (E8M0 scale per 32 elements), SiLU, no bias |
| quant_a1 | MXFP4 quant of the tokens x 7168 activations and their scatter into the per-expert, preshuffled A of gemm1 (two kernels) |
| gemm1 | per expert 256 x 4096 x 7168 (gate + up), epilogue SiLU + MXFP4 quant of the 256 x 2048 result |
| gemm2 | per expert 256 x 7168 x 2048, bf16 output per route (gather-reduce sums each token's 8 routes) |
| Config | token bucket 2048 of `aiter/configs/tuned_grouped_fmoe.csv` (1536 rounds up to the same row): tile 256x256x256, 2x2 waves, 4 LDS stages, 1x4 cluster; A preshuffled and B load hint 1 from the CSV. gemm1 and gemm2 are two specializations of one FlyDSL kernel (`launch_gemm_a8w4_tdm`), with the same tile settings |
| Schedule (env) | `AITER_FLYDSL_MMA_FIRST_GROUP=4 AITER_FLYDSL_MMA_GROUP=4 AITER_FLYDSL_DS_FIRST_N=8` (`mg4_dsfirst8` in the kernel names; 2489bae2d ran `mg1x2_dsfirst16`) |

How the work maps onto the GPU (one workgroup per WGP; the LDS stages leave no room for a second):

- gemm1 launches a worst-case grid of M tiles x 16 N tiles: 128 M tiles at 2048 tokens (112 at 1536). M tile *e*
  holds expert *e*'s rows (each expert starts on a 256-row boundary); the other 64 (48) M tiles have no rows and
  exit at once. So 1024 workgroups do work, about 4 rounds over the 256 WGPs.
- Each workgroup computes one 256 x 256 output tile of one expert. The 16 workgroups of an expert read the same A
  (256 rows x 7168; the 1x4 cluster multicasts it, the four clusters share it through L2). Each reads its own 256
  columns of the expert's W1 (896 KB), so **no weight byte is read twice** within a launch.
- gemm2: 64 experts x 28 N tiles = 1792 working workgroups (about 7 rounds), 8 K tiles each.
- Partial tiles (192 rows per expert at 1536 tokens): the A load's row extent is clamped to the expert's rows
  (TDM fills the rest of the LDS tile with zeros instead of reading it, per the kernel's comments), every WMMA still
  runs (25% of them on zero rows), gemm1's SiLU + quant skips 16-row blocks past the last row, and the TDM store
  writes only the valid rows. At 256 rows per expert the clamp equals the tile.

Kernels (names from the FlyDSL dump):
`a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg4_dsfirst8_regpart_planarlds_interleavelds_rolleddrain`
(gemm1) and the same with `_K2048_e64_cluster1x4_` (gemm2); quant_a1 = `moe_apre_quant_fd7168_r8_fp4_pk8_w16` +
`moe_apre_scatter_fd7168_r8_fp4_pk8_w16_tk7168_grid_sw8_async_ballot`.

## Setup

What the results below used (node heliosr-1b114-d01-3, one MI450X):

| Component | Version |
|---|---|
| Python | 3.12 |
| ROCm, PyTorch | `rocm[libraries,devel,device-gfx1250]==10.1.0a20260820`, `torch[device-gfx1250]==2.11.0+rocm10.1.0a20260820` from https://rocm.nightlies.amd.com/whl-multi-arch/ |
| FlyDSL | `flydsl-0.3.3.dev882-cp312-cp312-linux_x86_64.whl` in the repository root (what `run_a4w4_conc6144.sh` requires; `requirements.txt` pins 0.4.1, not used here) |
| AITER | this branch, installed without composable_kernel |
| Triton | AITER imports it; we used a source build of triton-lang/triton at 93035c44 (`3.9.0`) |
| Other Python packages | numpy, pandas, pyyaml, einops, psutil, ninja (AITER's JIT) |
| ROCm on the host | `/opt/rocm` (hipcc for AITER's JIT host modules; FlyDSL's `ld.lld`) |

Install AITER and FlyDSL into a separate directory (`--target`), so the venv keeps torch and Triton untouched:

```bash
cd <this repository>
SITE=$PWD/../moe_exp_site
pip install --no-deps --target $SITE ./flydsl-0.3.3.dev882-cp312-cp312-linux_x86_64.whl
AITER_USE_SYSTEM_TRITON=1 ENABLE_CK=0 PREBUILD_KERNELS=0 GPU_ARCHS=gfx1250 PYTHONPATH=$SITE \
    pip install --no-deps --no-build-isolation --target $SITE .
```

The first run JIT-compiles AITER's host modules (`module_aiter_core`, `module_moe_asm`) into `$SITE/aiter/jit/`
(a few minutes, once) and the FlyDSL kernels into the run's cache directory. `ENABLE_CK=0` must also be set at
run time (bench.sh does). FlyDSL 0.3.3.dev882 finds `ld.lld` only through `ROCM_PATH` (bench.sh defaults it to
`/opt/rocm`); without it compilation fails with "lld invocation failed". Never import `aiter` with the
repository root on `sys.path` (e.g. `python -c "import aiter"` from the root): the source tree then shadows the
site and JIT-builds into it.

## Running

From the repository root:

```bash
SITE=$PWD/../moe_exp_site op_tests/flydsl_tests/moe_exp/bench.sh            # gemm1, 2048 tokens, rotating; 3 rounds
# on a shared node: lock every GPU process; inside the lock check the GPU is idle before each process
LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5
# interleaved A/B: variant = name:ENV=VAL,ENV=VAL on top of run_a4w4_conc6144.sh's environment
SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 3 base bth0:AITER_TDM_B_TH=0 rot1:MOE_EXP_ROTATE=1
# the earlier benchmark: Yan's shape, all three kernels, no rotation
MOE_EXP_TOKENS=1536 MOE_EXP_KERNELS=all MOE_EXP_ROTATE=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5
# counter mode: HBM read bytes per gemm1 launch (rocprofv3 --pmc, public GL2C counters; no timing).
# Not proven: its only run hung the node, see below before using it.
PMC_ROCPROF=$VENV/bin/rocprofv3 LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 1 rot4 rot1:MOE_EXP_ROTATE=1
```

Each (variant, round) is one process running `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py` with
`run_a4w4_conc6144.sh`'s environment and arguments (tokens from `MOE_EXP_TOKENS`), through `bench_driver.py`. Round 0 is a warm-up/compile round
and is not counted. Results go to `moe_exp_runs/<UTC time>/` (or `OUT`): `meta.txt` (node, boot time, versions,
MEC/PM firmware, clock, power cap), one log per process, `out/*.pt` (outputs), `dump-<variant>/` (FlyDSL IR, named
after the kernels), and `summary.txt` from `summarize.py`. With `IDLE_CHECK=1` a process runs only if, inside the
lock, `rocm-smi` shows 0% GPU use and less than `IDLE_VRAM_MB` (1024) MB of VRAM in use; otherwise the series
stops. The VRAM check catches a loaded but idle model server, which shows 0% use.

Counter mode (`PMC_ROCPROF` = a rocprofv3 matching the venv's ROCm) runs each process under
`rocprofv3 --pmc GL2C_EA_RDREQ GL2C_EA_RDREQ_128B GL2C_READ_SECTORS GL2C_REQ` and replaces all timing with
`MOE_EXP_PMC_LAUNCHES` (8) spaced launches (synchronize + 2 ms idle before each) of gemm1, rotating as configured,
after a control of 4 copies of 1 GiB that is not in L2. `pmc_hbm.py` (run at the end, `pmc_summary.txt`) converts
`GL2C_EA_RDREQ` to bytes with the control's bytes per request and compares each launch with the input bytes. The
GL2C counters count without the XCD counter-clock register (they sit on the AID).

**This mode has never finished, and `pmc_hbm.py` has never read real output.** Its only run (2026-11-02 02:44 UTC,
d01-3, with the BPC23 voltage offset active) hung the node within about 2 minutes, during the first process's compile.
Every reader of the driver's power-management lock froze, including other users' tools, later dmesg showed
`SMU: No response`, and the node needed a power cycle. The cause is not proven. rocprofv3's preload also loads its tool
into FlyDSL's `ld.lld` child processes (each process compiles its kernels, so this happens in every process), and one of
those children was among the frozen tasks. On a shared node, ask before running it.

## What is measured

The test's kernel-bench has three launch callables, in this order: `quant_a1`, `gemm1`, `gemm2`. Each callable
is the Python launcher bound to one eager run's tensors, so calling it re-launches that stage. `bench_driver.py`
times only the kernels in `MOE_EXP_KERNELS` (default gemm1) and wraps each timed GEMM in a rotating launcher:

- **Rotation** (`MOE_EXP_ROTATE`, default 4): copy *i* of {W, W scales} serves launches *i*, *i*+4, *i*+8, ... Each
  copy holds a full copy of the buffer's storage (same offset and strides), so every copy feeds the kernel identical
  bytes and the outputs stay bit-identical. Between two uses of a copy the other three copies (1 GB of storage each at
  2048 tokens) stream through L2, far more than it holds (a 64 MiB ring stays resident in one AID's L2, 128 MiB does
  not: 2026-10-24 latency ladder), so every launch should read its weights from HBM. That is from these sizes and from
  timing (8 copies time the same as 4, see the results), not from counters. `MOE_EXP_ROTATE_A=1` rotates A and its
  scales the same way (A is still reused inside a launch, 16 N tiles per expert); `MOE_EXP_ROTATE=1` reuses one set of
  buffers, as before 2026-11-02. Both timers below rotate.

- **profiler_us**: the test's own number. For each callable AITER's `run_perftest(testGraph=False)` makes 5 warm-up
  calls, then (with `AITER_LOG_MORE=1`) 128 synchronized calls timed with events (logged, not used), then 128
  calls with no synchronization in between inside the torch profiler, and averages the kernels' device durations
  from the trace (first iteration and outliers outside 1.5 IQR dropped; a two-kernel callable like `quant_a1`
  reports the sum). The callables pass no tensor arguments, so `run_perftest`'s own input rotation does nothing;
  `bench_driver.py`'s rotating launcher does the rotation. `bench_driver.py` prints each timed kernel's value as
  `[moe-exp] <name>: profiler_us`. If it prints
  `n/a` (the test: `us = nan`), the profiler trace had no kernel events; `profiler_probe.py` checks whether the
  node's torch profiler records kernels at all (on d01-3 it stopped after warm reboots; see the 2026-11-01 results).
- **graph_us**: after the test's timing, `bench_driver.py` captures each timed callable 128 times (a multiple of
  the rotation count, so the copies keep their order across replays) into one CUDA graph, replays it 10 times, and times each replay with CUDA events: the median time per launch.
  No profiler; launches run back to back and the number includes the ~1.5 us gaps between kernels. The GPU idles
  1 s before each callable's graph timing (`MOE_EXP_GRAPH_GAP_S`), so one kernel's back-to-back run does not heat
  the GPU for the next kernel.
- **Correctness**: the test compares the fused-MoE output with its torch reference (`logits_diff`, gate 0.01)
  and raises otherwise; `summarize.py` also compares every saved output bit for bit with the first variant's
  round-1 output.
- **TB/s**: bytes / time, each byte counted once (`summarize.py`): gemm1 = A + W1 + MXFP4 output, each with its
  E8M0 scales = 1,078.5 MB at 2048 tokens (1,058.4 MB at 1536); gemm2 = A2 + W2 (+ scales) + bf16 output = 688.7 MB
  at 1536 tokens. The test's own "Bandwidth" counts gemm1's output as bf16, the grouped-MoE tuner's convention.

Why the two timers differ. They time different launch regimes, and only one of them traces:
- The Python launch path takes about 190 us per call (the `AITER_LOG_MORE` event lines: ~272 us per synchronized
  gemm1 launch for an 84 us kernel), longer than either GEMM. So in the profiler loop the host probably cannot keep
  the GPU fed and each launch starts on an idle GPU (inferred, not measured). The graph replays run back to back.
- Back to back, a power-limited GPU lowers its clock. On 2026-10-29 at a 2.5 kW socket limit (this node), 2.5 s
  of back-to-back gemm1 ran at about 1.07 GHz against about 2.2 GHz for spaced launches. A kernel that saves
  cycles but spends more energy per launch can be faster on profiler_us and slower on graph_us; 8ea2265dc's gemm1
  did exactly that against 2489bae2d (-6.5% and +4.1%).
- The profiler adds a timestamp step per dispatch. Its cost depends on the MEC firmware (about 4-5 us per launch
  on build 2430; build 882532 inflated memory-bound kernels 1.5-2.3x).
- Back-to-back launches do not overlap: a probe kernel that stamped every workgroup's start and end (2026-10-25)
  never saw a workgroup start before the previous kernel's last one ended. (This README said the opposite until
  2026-11-01.)

Rules we follow for numbers worth comparing: one shape per process; interleave the variants being compared;
median of 3 or more timed rounds; compare only numbers from the same boot; keep `meta.txt` with the results.
Clocks, power limits and firmware matter: a node whose perf settings were not re-applied after a reboot can run
5-20% slow. After a reboot also run `profiler_probe.py`: if it gets no events, profiler_us stays n/a on that boot.

## Results

### 2026-11-02, gemm1 at 2048 tokens with rotation, heliosr-1b114-d01-3 (MI450X), at 8ea2265dc, MEC 2430, BPC23 perf setup applied

`results/20261102_d01-3_mec2430_bpc23_g1rot/` (meta.txt, summary.txt, run_log.txt, one full process log,
series1_summary.txt, profiler_probe.txt). 5 timed rounds, one process each, interleaved, 17:59-18:11 UTC. Command with
the kit's current defaults: `LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5
rot4a:MOE_EXP_ROTATE_A=1 rot8a:MOE_EXP_ROTATE=8,MOE_EXP_ROTATE_A=1 rot4 rot1:MOE_EXP_ROTATE=1`. In the saved files
A rotation was still the default, so these variants are named `rot4`, `rot8`, `rot4w` and `rot1`.

| Inputs rotated | profiler_us (median, range) | graph_us (median, range) | TB/s (profiler) | TB/s (graph) | graph_us vs none |
|---|---|---|---|---|---|
| None | 78.31 (77.62-79.22) | 78.39 (78.25-78.54) | 13.77 | 13.76 | - |
| **Weights, 4 copies (default)** | **82.06** (81.11-82.49) | **82.06** (82.04-82.18) | **13.14** | **13.14** | +4.7% |
| Weights + A, 4 copies | 83.23 (82.87-83.73) | 84.38 (84.17-84.66) | 12.96 | 12.78 | +7.6% |
| Weights + A, 8 copies | 83.63 (83.06-85.08) | 84.34 (84.30-84.49) | 12.90 | 12.79 | +7.6% |

- Node state: boot 2026-11-02 14:24 UTC after a power cycle; MEC firmware 2430, PM 02.125.15.00, MAX_CLK 2400 MHz,
  2.5 kW socket power limit. `perf_setup_BPC23.sh` applied at 16:02 UTC: voltage offset -82 mV, CSC DVO offset 25 mV, CPU
  governor performance, NUMA balancing off, amdgpu default parameters. REALTIME - KFD clock +0.017 ms;
  `profiler_probe.py`: 256 of 256 events.
- Correct in every round (`logits_diff=3.4257e-06`, `rel_l2=2.6175e-03`); all 24 outputs bit-identical.
- A separate series (17:53-17:58 UTC, `series1_summary.txt`) reproduces weights + A, 4 copies (profiler 84.06, graph
  84.27) and no rotation (profiler 77.47, graph 78.38).
- 8 copies time the same as 4 (graph -0.05%), so more distance between two uses of a copy gains nothing: 4 copies leave
  nothing reusable behind. This is timing evidence. HBM read bytes were not counted, because counter mode hung the node
  (see Running).
- Without rotation, launches are 3.7 us faster from reused weights and another 2.3 us from reused A (graph_us).
  Probably data left in L2 by the previous launch; inferred from timing.
- Against the same kernels at 1536 tokens without rotation (74.96 us, 14.12 TB/s on 2026-11-01; another boot, so a rough
  comparison), 2048 tokens without rotation takes 4.6% longer for 1.9% more bytes.

### 2026-11-02, heliosr-1b114-d01-3 (MI450X), at 8ea2265dc, MEC 2430, BPC23 perf setup applied

`results/20261102_d01-3_mec2430_bpc23_base/` (meta.txt, summary.txt, run_log.txt, one full process log,
profiler_probe.txt). Command: `LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5` with the
kit at f471dc7f4, 5 timed rounds, one process each, 00:42-00:45 UTC.

| Kernel | profiler_us (median, range) | graph_us (median, range) | TB/s (profiler) | TB/s (graph) |
|---|---|---|---|---|
| quant_a1 | 21.73 (21.65-21.89) | 20.33 (20.26-20.38) | - | - |
| gemm1 | 74.63 (73.73-74.97) | 74.96 (74.81-75.04) | 14.18 | 14.12 |
| gemm2 | 52.22 (51.67-52.52) | 51.80 (51.75-51.99) | 13.19 | 13.29 |

- Node state: boot 2026-11-01 23:42 UTC after a power cycle; MEC firmware 2430 (sha256 d798bf71...), PM 02.125.15.00,
  MAX_CLK 2400 MHz, 2.5 kW socket power limit. `perf_setup_BPC23.sh` v2026.10.29 applied at 00:25 UTC: voltage
  offset -82 mV, CSC DVO offset 25 mV, CPU governor performance, NUMA balancing off, amdgpu default parameters
  (MTYPE RW, XNACK off). Its `perf_env.sh` was not sourced: it points `ROCM_PATH` and `LD_LIBRARY_PATH` at
  /opt/rocm instead of the venv's ROCm, and its only non-default setting is `HSA_XNACK=1`.
- The torch profiler records kernels again on this boot (`profiler_probe.py`: 256 of 256 events), so profiler_us is
  back.
- Correct in every round (`logits_diff=3.4249e-06`, `rel_l2=2.6172e-03`); all 6 outputs bit-identical.
- Against 2026-11-01 (same code and firmware; another boot, perf settings not applied): graph_us +0.4% quant_a1,
  +0.3% gemm1, +0.2% gemm2, so no measurable change from the perf setup. Different boots, so a rough comparison.
- profiler_us and graph_us agree within 0.8% (gemm1 74.63 vs 74.96, gemm2 52.22 vs 51.80). On 2026-10-28
  (2489bae2d, MEC 2560) profiler_us was 3.6% (gemm1) and 9.6% (gemm2) above graph_us.

### 2026-11-01, heliosr-1b114-d01-3 (MI450X), at 8ea2265dc

`results/20261101_d01-3_base/` (meta.txt, summary.txt, run_log.txt, one full process log, profiler_probe.txt).
Command: `LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5`, 5 timed rounds, one process
each, 21:15-21:21 UTC.

| Kernel | graph_us (median, range) | TB/s (graph) | profiler_us |
|---|---|---|---|
| quant_a1 | 20.25 (20.11-20.28) | - | n/a |
| gemm1 | 74.76 (74.69-74.95) | 14.16 | n/a |
| gemm2 | 51.69 (51.63-51.74) | 13.32 | n/a |

- **No profiler_us on this boot.** The torch profiler returned no device events for gemm1 and gemm2 in any round
  (the test prints `us = nan`) and a single, implausible event (2.85 ms) for quant_a1 in 2 of 5 rounds.
  `profiler_probe.py` gets 0 of 256 events for plain torch copies and matmuls on the same boot, while rocprofv3
  `--kernel-trace` records every kernel. So the node's torch-profiler path is broken, not this code; on
  2026-10-29 (MEC 2560) the same code gave profiler numbers. Cause, found the same day: the GPU's REALTIME counter,
  which stamps kernel completion, ran 38.8 s ahead of KFD's clock counter, which ROCr uses to convert GPU ticks to
  host time, so every kernel record landed outside the profiler's window. Each warm reboot added another 38.8 s
  (77.7 s after the next one); a power cycle re-aligned them (2026-11-02 results).
- Correct in every round (`logits_diff=3.4249e-06`, `rel_l2=2.6172e-03`); all 6 outputs bit-identical.
- Node state: boot 2026-11-01 08:28 UTC, MEC firmware 2430 (`gc_12_1_0_mec.bin` sha256 d798bf71...), PM
  02.125.15.00, MAX_CLK 2400 MHz, 2.5 kW socket power limit, the node's perf settings not re-applied after that boot.
- Against 2026-10-29 (same code; MEC 2560, another boot, the kit's first driver): graph_us gemm1 84.83 -> 74.76
  (-11.9%), gemm2 54.18 -> 51.69 (-4.6%), quant_a1 20.73 -> 20.25. Firmware, boot and the graph timer's order
  changed together, so the cause is not known.

### 2026-10-28, heliosr-1b114-d01-3 (MI450X), at 2489bae2d (`run_dsr1.sh`'s environment)

Measured on the branch's earlier base, now `dev/lixzhang/flydsl_moe_exp_2489bae2d`, with the kit's first version.
That version graph-timed each kernel right after its own profiler loop, so gemm2's profiler loop ran right after
gemm1's back-to-back replays. 2489bae2d's gemm2 is insensitive to that (about 59 us with and without the driver);
8ea2265dc's is not (54.2 us cool, 60.5 us after gemm1's replays, 2026-10-29), hence the change.
`results/20261028_d01-3_base/` (meta.txt, summary.txt, run_log.txt, one full process log). Command:
`LOCK=gpu-lock IDLE_CHECK=1 SITE=... op_tests/flydsl_tests/moe_exp/bench.sh 5`, 5 timed rounds, one process each.

| Kernel | profiler_us (median, range) | graph_us (median, range) | TB/s (profiler) | TB/s (graph) |
|---|---|---|---|---|
| gemm1 | 83.76 (83.73-84.30) | 80.85 (80.35-81.05) | 12.64 | 13.09 |
| gemm2 | 58.87 (58.40-59.30) | 53.73 (53.62-54.05) | 11.70 | 12.82 |

- Correct in every round (`logits_diff=3.4249e-06`, `rel_l2=2.6172e-03`); all 6 outputs bit-identical.
- A run 5 minutes earlier gave 83.43 / 80.70 / 59.33 / 53.68 us (same order as the table).
- Node state: boot 2026-10-28 14:09 UTC, MEC firmware 2560 (not the stock 2430), PM 02.125.15.00, MAX_CLK
  2400 MHz, 2.5 kW socket power limit, the node's perf settings not re-applied after that boot.

Earlier the same day, same boot and code (previous harness, profiler timer, 3 interleaved rounds), relative to
the base: `AITER_TDM_B_TH=0` (temporal B loads) +10.3% gemm1 / +5.0% gemm2; `AITER_FLYDSL_MXFP4_CLUSTER_M=4`
(4x4 cluster for both GEMMs) +6.3% / +6.8%.

## Notes

- B load hint: in 8ea2265dc `AITER_TDM_B_TH` overrides the config's `tdm_b_th` column only when set
  (`aiter/ops/flydsl/grouped_gemm_mxfp4.py`, `_select_tdm_b_th`). In 2489bae2d it defaulted to 1 and always won.
- `waves_per_tensor_tdm=4` gives wrong results (about 60% NaN) at 2489bae2d: with 2 M waves the A-scale is split
  along K. A fix is on `dev/lixzhang/yadai_dsr1_tdm_scale_waves` (`AITER_FLYDSL_TDM_SCALE_WAVES=2`). Not rechecked
  at 8ea2265dc.
- B (weights) is never shared at this shape, and a 4x4 cluster puts four different experts in a column. The
  kernel multicasts B only between M tiles of the same expert, which needs more than 256 rows per expert
  (more than 2048 tokens at 64 experts, top-8).
- `quant_a1` at 8ea2265dc (`moe_apre_quant_*_pk8_w16` + `moe_apre_scatter_*`) took 20.7 us graph_us on 2026-10-29,
  against 14.0 us for 2489bae2d's `moe_apre_gather_*` + `moe_apre_quant_*_qks_k512_pkamax` (#6984).
