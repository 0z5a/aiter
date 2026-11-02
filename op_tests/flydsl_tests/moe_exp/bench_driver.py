# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run test_flydsl_grouped_gemm.py's main() unchanged, with these additions.

1. Kernel selection: ``MOE_EXP_KERNELS`` (comma list of quant_a1, gemm1, gemm2, or
   ``all``; default ``gemm1``) are the launch callables that get timed. The test
   still runs the whole MoE once and checks it against its torch reference; the
   other callables are not timed (the test then prints ``us = nan`` and a
   "timing unavailable" warning for them).
2. Rotating inputs: ``MOE_EXP_ROTATE=N`` (default 4) copies of each timed GEMM's
   weights and weight scales, used in turn by consecutive launches, so a launch
   never finds its weights in a cache: between two uses of one copy, N-1 launches
   stream the other copies (about 1 GB each) through L2. ``MOE_EXP_ROTATE_A=1``
   (default 0) rotates the GEMM's A and A scales the same way, so all inputs come
   from HBM at least once per launch (A reuse inside a launch, across the N tiles
   of an expert, is part of the kernel). Each copy holds a full copy of the
   buffer's storage at the same offset and strides, so the kernel sees identical
   bytes. N=1 = no rotation (the earlier benchmark).
3. ``[moe-exp] <name>: profiler_us = ...``: the test's own run_perftest value
   (torch profiler, eager launches) for each timed callable, rotating.
4. A profiler-free kernel time: after the test's timing, each timed callable is
   captured ``MOE_EXP_GRAPH_ITERS`` times (a multiple of N) into one CUDA graph,
   rotating, the graph is replayed ``MOE_EXP_GRAPH_REPS`` times and each replay
   is timed with CUDA events: ``[moe-exp] <name>: graph_us = <median per
   launch>``, including the gaps between back-to-back kernels. The GPU idles
   ``MOE_EXP_GRAPH_GAP_S`` seconds (default 1) before each callable's graph
   timing.
5. Counter mode: ``MOE_EXP_PMC_LAUNCHES=L`` (> 0) replaces all timing with L
   spaced launches (synchronize + 2 ms idle before each) of each selected
   callable, rotating, for ``rocprofv3 --pmc`` (``pmc_hbm.py``). It first runs a
   control: 4 copies of a 1 GiB buffer (two sources, so the source is never in
   L2), whose read bytes are known.
6. ``MOE_EXP_SAVE_OUT=<file.pt>`` saves the fused-MoE output that the test checks
   against its torch reference, for a bit-exact comparison between runs.

usage (from the repository root):
    python op_tests/flydsl_tests/moe_exp/bench_driver.py <test_flydsl_grouped_gemm.py args>
"""

import functools
import importlib.util
import os
import statistics
import sys
import time

import torch

from aiter import test_common

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(os.path.dirname(HERE), "test_flydsl_grouped_gemm.py")
ALL_KERNELS = ("quant_a1", "gemm1", "gemm2")
_k = os.environ.get("MOE_EXP_KERNELS", "gemm1").strip()
KERNELS = ALL_KERNELS if _k in ("", "all") else tuple(_k.split(","))
ROTATE = int(os.environ.get("MOE_EXP_ROTATE", "4"))
ROTATE_A = os.environ.get("MOE_EXP_ROTATE_A", "0") == "1"
GRAPH_ITERS = int(os.environ.get("MOE_EXP_GRAPH_ITERS", "128"))
GRAPH_REPS = int(os.environ.get("MOE_EXP_GRAPH_REPS", "10"))
GRAPH_GAP_S = float(os.environ.get("MOE_EXP_GRAPH_GAP_S", "1"))
PMC_LAUNCHES = int(os.environ.get("MOE_EXP_PMC_LAUNCHES", "0"))
SAVE_OUT = os.environ.get("MOE_EXP_SAVE_OUT")
for _name in KERNELS:
    if _name not in ALL_KERNELS:
        raise SystemExit(f"MOE_EXP_KERNELS: unknown kernel {_name!r} (use {ALL_KERNELS} or all)")
if ROTATE < 1:
    raise SystemExit("MOE_EXP_ROTATE must be >= 1")
if GRAPH_ITERS % ROTATE:
    raise SystemExit(f"MOE_EXP_GRAPH_ITERS={GRAPH_ITERS} must be a multiple of MOE_EXP_ROTATE={ROTATE}")

_timed = []  # (name, rotating callable) per timed kernel, in run_perftest call order

# Positional arguments of flydsl_grouped_gemm_a8w4_masked(out, a, w, a_scales, w_scales, m_tile_map, ...)
_A, _W, _AS, _WS = 1, 2, 3, 4


def kernel_name(func):
    """Which stage a kernel-bench launch callable (a functools.partial) runs."""
    fn = getattr(func, "func", None)
    kw = getattr(func, "keywords", None) or {}
    name = getattr(fn, "__name__", "")
    if name == "flydsl_grouped_gemm_a8w4_masked":
        return "gemm1" if (kw.get("stage1_act") or kw.get("gemm_stage") == 1) else "gemm2"
    if name == "flydsl_moe_fused_quant_preshuffle":
        return "quant_a1" if kw.get("topids_to_rows") is not None else "quant_a2"
    return None


def storage_copy(t):
    """A tensor like ``t`` (dtype, shape, strides, storage offset) over a new copy of its whole storage."""
    src = t.untyped_storage()
    dst = torch.empty(src.nbytes(), dtype=torch.uint8, device=t.device).untyped_storage()
    dst.copy_(src)
    out = torch.empty(0, dtype=t.dtype, device=t.device)
    out.set_(dst, t.storage_offset(), t.size(), t.stride())
    return out


class Rotating:
    """Call ``calls[i % n]`` on the i-th call."""

    def __init__(self, calls):
        self.calls = calls
        self.i = 0

    def __call__(self):
        f = self.calls[self.i % len(self.calls)]
        self.i += 1
        return f()


def rotating(name, func):
    if ROTATE == 1 or not name.startswith("gemm"):
        return Rotating([func]), 0
    idx = [_W, _WS] + ([_A, _AS] if ROTATE_A else [])
    calls, nbytes = [func], 0
    for _ in range(ROTATE - 1):
        args = list(func.args)
        for i in idx:
            args[i] = storage_copy(args[i])
        calls.append(functools.partial(func.func, *args, **func.keywords))
    nbytes = sum(func.args[i].untyped_storage().nbytes() for i in idx)
    torch.cuda.synchronize()
    return Rotating(calls), nbytes


def graph_time_us(fn, iters, reps):
    """Median per-launch time of ``iters`` launches of ``fn`` in one CUDA graph."""
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2 * ROTATE):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)
    del graph
    torch.cuda.synchronize()
    return statistics.median(times), times


def spaced_launches(fn, n):
    for _ in range(n):
        torch.cuda.synchronize()
        time.sleep(0.002)
        fn()
    torch.cuda.synchronize()


_pmc_control_done = False


def pmc_control():
    """4 copies of 1 GiB, alternating two sources: each copy reads 1 GiB that is not in L2."""
    srcs = [torch.ones(1 << 28, device="cuda") for _ in range(2)]
    dst = torch.empty_like(srcs[0])
    for i in range(4):
        torch.cuda.synchronize()
        time.sleep(0.002)
        dst.copy_(srcs[i % 2])
    torch.cuda.synchronize()
    print("[moe-exp] pmc control: 4 x copy of 1 GiB (1,073,741,824 B read each)", flush=True)
    del srcs, dst
    torch.cuda.empty_cache()


_orig_run_perftest = test_common.run_perftest


def run_perftest(func, *args, **kwargs):
    global _pmc_control_done
    # The kernel-bench loop passes a bare launch callable with testGraph=False.
    if args or kwargs.get("testGraph", False):
        return _orig_run_perftest(func, *args, **kwargs)
    name = kernel_name(func) or f"callable{len(_timed)}"
    if name not in KERNELS:
        print(f"[moe-exp] {name}: not timed (MOE_EXP_KERNELS={','.join(KERNELS)})", flush=True)
        return None, float("nan")
    rot, nbytes = rotating(name, func)
    if nbytes:
        what = "W, W scales" + (", A, A scales" if ROTATE_A else "")
        print(
            f"[moe-exp] {name}: rotating {ROTATE} copies of {what} "
            f"({nbytes / 1e6:.1f} MB per copy)",
            flush=True,
        )
    if PMC_LAUNCHES > 0:
        if not _pmc_control_done:
            pmc_control()
            _pmc_control_done = True
        spaced_launches(rot, PMC_LAUNCHES)
        print(f"[moe-exp] {name}: {PMC_LAUNCHES} spaced launches for counters", flush=True)
        return None, float("nan")
    _timed.append((name, rot))
    return _orig_run_perftest(rot, **kwargs)


test_common.run_perftest = run_perftest

sys.path.insert(0, os.path.dirname(TEST))
spec = importlib.util.spec_from_file_location("test_flydsl_grouped_gemm", TEST)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
_orig_run = mod._run_grouped_via_fused_moe


def _run_grouped_via_fused_moe(*args, **kwargs):
    first = len(_timed)
    out, ref, us, kernel_us = _orig_run(*args, **kwargs)
    timed = _timed[first:]
    for name, _ in timed:
        if name in (kernel_us or {}):
            v = float(kernel_us[name])
            # run_perftest returns 0 when the profiler trace has no events for the callable's kernels
            text = f"{v:.2f}" if v > 0 else "n/a (no kernel events in the profiler trace)"
            print(f"[moe-exp] {name}: profiler_us = {text}", flush=True)
    if GRAPH_ITERS > 0 and PMC_LAUNCHES == 0:
        for name, func in timed:
            torch.cuda.synchronize()
            time.sleep(GRAPH_GAP_S)
            try:
                median, times = graph_time_us(func, GRAPH_ITERS, GRAPH_REPS)
            except RuntimeError as exc:  # report and keep the test's own result
                print(f"[moe-exp] {name}: graph timing failed: {exc!r}", flush=True)
                continue
            print(
                f"[moe-exp] {name}: graph_us = {median:.2f} "
                f"(median of {len(times)} replays x {GRAPH_ITERS} launches, "
                f"range {min(times):.2f}-{max(times):.2f}, after {GRAPH_GAP_S:g} s idle)",
                flush=True,
            )
    if SAVE_OUT:
        torch.save(out.detach().cpu(), SAVE_OUT)
        print(f"[moe-exp] saved output {SAVE_OUT} {tuple(out.shape)}", flush=True)
    return out, ref, us, kernel_us


mod._run_grouped_via_fused_moe = _run_grouped_via_fused_moe
print(
    f"[moe-exp] kernels {','.join(KERNELS)}; rotate {ROTATE}"
    f"{' (W and A)' if ROTATE_A else ' (W only)'}"
    + (f"; counter mode, {PMC_LAUNCHES} launches" if PMC_LAUNCHES else ""),
    flush=True,
)
sys.argv = [TEST] + sys.argv[1:]
mod.main()
