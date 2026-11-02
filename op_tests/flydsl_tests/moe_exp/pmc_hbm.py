# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""HBM read bytes per gemm1 launch, from a counter-mode bench.sh run (PMC_ROCPROF set).

Reads every ``pmc-<variant>_r<round>/**/*counter_collection.csv`` in a bench.sh output
directory. Per process it takes the last ``MOE_EXP_PMC_LAUNCHES`` gemm1 dispatches
(the driver's spaced, rotating launches) and the 4 copy dispatches right before them
(the control: each reads 1 GiB that is not in L2), and prints per gemm1 launch:

  GL2C_EA_RDREQ x (bytes per request)   L2 -> memory read bytes (HBM, or a cache below L2)
  GL2C_READ_SECTORS x 32                bytes the L2 served to its clients (hits + misses)

Bytes per EA request come from the control (1 GiB / its EA_RDREQ); GL2C_EA_RDREQ_128B
shows whether all requests are 128 B. The expected input bytes are summarize.py's byte
model without the output: A + A scales + W1 + W1 scales, each counted once.

usage: python pmc_hbm.py <bench.sh OUT directory>
"""

import collections
import csv
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import arg, read  # noqa: E402

COPY_BYTES = 1 << 30


def dispatches(csv_path):
    d = collections.OrderedDict()
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            key = int(r["Dispatch_Id"])
            e = d.setdefault(key, {"name": r["Kernel_Name"], "c": {}})
            e["c"][r["Counter_Name"]] = e["c"].get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
    return [d[k] for k in sorted(d)]


def main(d):
    meta = read(os.path.join(d, "meta.txt"))
    args = re.search(r"^args\s+(.*)$", meta, re.MULTILINE).group(1)
    launches = int(re.search(r"^pmc\s+(\d+) launches", meta, re.MULTILINE).group(1))
    E, T, topk = int(arg(args, "experts")), int(arg(args, "tokens")), int(arg(args, "topk"))
    D, inter = int(arg(args, "model-dim")), int(arg(args, "inter-dim"))
    R = T * topk
    fp4 = lambda n: n // 2 + n // 32  # noqa: E731
    inputs = fp4(R * D) + fp4(E * 2 * inter * D)
    print(meta.rstrip())
    print(f"\nexpected gemm1 input bytes (A + scales + W1 + scales, once each): {inputs / 1e6:.1f} MB\n")
    hdr = (f"{'process':12} {'B/EA req':>8} {'128B share':>10} {'EA read MB (per launch, median)':>32} "
           f"{'vs inputs':>9} {'L2 served MB':>12}")
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(glob.glob(os.path.join(d, "pmc-*", "**", "*counter_collection.csv"), recursive=True)):
        proc = os.path.relpath(p, d).split(os.sep)[0][4:]
        ds = dispatches(p)
        g1 = [i for i, e in enumerate(ds) if "a8w4_tdm" in e["name"] and "_act1_" in e["name"]]
        if len(g1) < launches:
            print(f"{proc:12} only {len(g1)} gemm1 dispatches, expected {launches}")
            continue
        spaced = g1[-launches:]
        copies = [i for i, e in enumerate(ds[: spaced[0]]) if "copyBuffer" in e["name"]][-4:]
        if len(copies) < 4:
            print(f"{proc:12} control copies not found")
            continue
        ea_copy = statistics.median(ds[i]["c"]["GL2C_EA_RDREQ"] for i in copies)
        bpr = COPY_BYTES / ea_copy
        share = statistics.median(ds[i]["c"]["GL2C_EA_RDREQ_128B"] / ds[i]["c"]["GL2C_EA_RDREQ"] for i in spaced)
        ea = [ds[i]["c"]["GL2C_EA_RDREQ"] * bpr for i in spaced]
        served = [ds[i]["c"]["GL2C_READ_SECTORS"] * 32 for i in spaced]
        med = statistics.median(ea)
        cells = f"{med / 1e6:.1f} ({min(ea) / 1e6:.1f}-{max(ea) / 1e6:.1f})"
        print(f"{proc:12} {bpr:8.1f} {share:10.3f} {cells:>32} {med / inputs:9.3f} "
              f"{statistics.median(served) / 1e6:12.1f}")
        print(f"{'':12} per launch EA MB: {' '.join(f'{x / 1e6:.0f}' for x in ea)}")


if __name__ == "__main__":
    main(sys.argv[1])
