#!/usr/bin/env python3
"""Compare public-API CPU benchmark CSVs; reject shape/checksum mismatches."""
import argparse
import csv
import math
import sys


def read(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dtype"], int(row["k"]), int(row["m"]), int(row["n"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate case {key}")
            ms = float(row["median_ms"])
            checksum = float(row["checksum"])
            if not math.isfinite(ms) or ms <= 0 or not math.isfinite(checksum):
                raise ValueError(f"{path}: invalid measurement {key}")
            rows[key] = (ms, checksum)
    if not rows:
        raise ValueError(f"{path}: no measurements")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    baseline, candidate = read(args.baseline), read(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline/candidate cases differ")
    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(["dtype", "k", "m", "n", "baseline_ms", "candidate_ms", "speedup", "checksum_equal"])
    for key, (before, checksum) in baseline.items():
        after, candidate_checksum = candidate[key]
        if checksum != candidate_checksum:
            raise ValueError(f"checksum differs for {key}: {checksum} vs {candidate_checksum}")
        out.writerow([*key, f"{before:.6f}", f"{after:.6f}", f"{before / after:.3f}", True])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"compare_cpu_bench: {error}", file=sys.stderr)
        sys.exit(1)
