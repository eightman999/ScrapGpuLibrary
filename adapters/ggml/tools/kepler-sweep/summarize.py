#!/usr/bin/env python3
"""kepler-sweep: parsers and the summary of the CSV files that sweep.sh writes.

  summarize.py test    <log>             -> "<passed> <total>" of the last "N/M tests passed" line of a test-backend-ops log
  summarize.py perf    <log> <gpu> <variant>   -> csv rows gpu,variant,type,m,n,k,gflops of a `test-backend-ops perf` log (colors stripped)
  summarize.py bench   <llama-bench csv>  -> "<pp t/s> <tg t/s>"
  summarize.py kernels <stderr log>       -> kernels per decode token (GGML_CL11_LAUNCH_LOG) or "-"
  summarize.py best    <mm.csv> <gpu> <n> <count> -> the names of the best variants for the GPU (geometric mean of the speed-up over `base`
                                                    of the quantized types at that n), one per line; empty when there is no data
  summarize.py mvgbs   <mm_bench csv> <gpu> <config> -> rows gpu,config,type,m,k,ms,gbs (bytes of the weights / time of an n=1 mat-vec)
  summarize.py kl      <llama-perplexity log>  -> "ppl,mean_kld,kld99,kld999,max_dp,same_top_p,verdict" (verdict against KL_MEAN_MAX 0.0070,
                                                  KL_P99_MAX 0.033, KL_TOP_MIN 95.0; empty fields when the log has no statistics)
  summarize.py summary <out dir>          -> writes <out dir>/summary.md from the csv files
"""
import csv
import math
import os
import re
import sys
from collections import defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def cmd_test(path):
    last = None
    for line in open(path, errors="replace"):
        m = re.search(r"(\d+)/(\d+) tests passed", ANSI.sub("", line))
        if m:
            last = m
    print(f"{last.group(1)} {last.group(2)}" if last else "0 0")


def cmd_perf(path, gpu, variant):
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        m = re.search(r"type_a=(\w+),type_b=\w+,m=(\d+),n=(\d+),k=(\d+),.*?([0-9.]+)\s+([GTMK])FLOPS", line)
        if not m:
            continue
        scale = {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}[m.group(6)]
        q = '"' + variant.replace('"', '""') + '"'   # variant names may contain commas ("x88,ku=4")
        print(f"{gpu},{q},{m.group(1)},{m.group(2)},{m.group(3)},{m.group(4)},{float(m.group(5)) * scale:.1f}")


def cmd_bench(path):
    pp = tg = "-"
    with open(path, errors="replace") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        try:
            if int(r.get("n_prompt", 0)) > 0 and int(r.get("n_gen", 0)) == 0:
                pp = f"{float(r['avg_ts']):.1f}"
            if int(r.get("n_gen", 0)) > 0 and int(r.get("n_prompt", 0)) == 0:
                tg = f"{float(r['avg_ts']):.1f}"
        except (KeyError, ValueError):
            pass
    print(pp, tg)


def cmd_kernels(path):
    counts = defaultdict(int)
    for line in open(path, errors="replace"):
        m = re.search(r"cl11 launches: nodes (\d+), mul_mat (\d+) \(max columns (\d+)\), kernels (\d+)", line)
        if m and m.group(3) == "1":
            counts[int(m.group(4))] += 1
    print(max(counts, key=counts.get) if counts else "-")


BPW = {"q4_0": 18 / 32, "q8_0": 34 / 32, "q4_k": 144 / 256, "q6_k": 210 / 256, "f16": 2.0, "f32": 4.0}  # bytes per weight


def cmd_mvgbs(path, gpu, config):
    if not os.path.exists(path):
        return
    for line in open(path, errors="replace"):
        p = line.strip().split(",")
        if len(p) < 7 or p[0] not in BPW:
            continue
        try:
            m, k, ms = int(p[1]), int(p[3]), float(p[5])
        except ValueError:
            continue
        if ms > 0:
            print(f"{gpu},{config},{p[0]},{m},{k},{ms:.4f},{m * k * BPW[p[0]] / (ms * 1e-3) / 1e9:.2f}")


def cmd_kl(path):
    text = ANSI.sub("", open(path, errors="replace").read()) if os.path.exists(path) else ""

    def num(pat):
        m = re.search(pat, text)
        return m.group(1) if m else ""

    ppl = num(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)")
    mean = num(r"Mean\s+KLD:\s*([0-9.eE+-]+)")
    p99 = num(r"99\.0%\s+KLD:\s*([0-9.eE+-]+)")
    p999 = num(r"99\.9%\s+KLD:\s*([0-9.eE+-]+)")
    maxdp = num(r"Maximum\s+\S+p:\s*([0-9.]+)")
    top = num(r"Same top p:\s*([0-9.]+)")
    verdict = "-"
    if mean and p99 and top:
        lim_mean = float(os.environ.get("KL_MEAN_MAX", "0.0070"))
        lim_p99 = float(os.environ.get("KL_P99_MAX", "0.033"))
        lim_top = float(os.environ.get("KL_TOP_MIN", "95.0"))
        verdict = "PASS" if float(mean) <= lim_mean and float(p99) <= lim_p99 and float(top) >= lim_top else "FAIL"
    print(",".join([ppl, mean, p99, p999, maxdp, top, verdict]))


def read_mm(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    return rows


def speedups(rows, gpu, n):
    """variant -> list of speed-ups over base for the quantized types at n columns"""
    base = {}
    for r in rows:
        if r["gpu"] == gpu and r["variant"] == "base" and r["n"] == str(n):
            base[(r["type"], r["m"], r["k"])] = float(r["gflops"])
    sp = defaultdict(list)
    for r in rows:
        if r["gpu"] != gpu or r["variant"] == "base" or r["n"] != str(n) or not r["type"].startswith("q"):
            continue
        b = base.get((r["type"], r["m"], r["k"]))
        if b and float(r["gflops"]) > 0:
            sp[r["variant"]].append(float(r["gflops"]) / b)
    return sp


def cmd_best(path, gpu, n, count):
    sp = speedups(read_mm(path), gpu, n)
    ranked = sorted(sp.items(), key=lambda kv: -math.exp(sum(math.log(x) for x in kv[1]) / len(kv[1])))
    for name, _ in ranked[:int(count)]:
        print(name)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def cmd_summary(d):
    md = ["# kepler-sweep summary", "", f"directory: {d}", ""]

    # correctness
    tests = read_mm(os.path.join(d, "test.csv"))
    if tests:
        md += ["## test-backend-ops / graph checks", "", table([[r["gpu"], r["config"], r["passed"], r["total"], "PASS" if r["passed"] == r["total"] and r["total"] != "0" else "FAIL"] for r in tests],
                                                              ["GPU", "config", "passed", "total", "result"]), ""]
    greedy = read_mm(os.path.join(d, "greedy.csv"))
    if greedy:
        md += ["## greedy 64 tokens", "", table([[r["gpu"], r["model"], r["config"], r["identical"], r["kernels_per_token"]] for r in greedy],
                                               ["GPU", "model", "config", "identical to the reference", "kernels per decode token"]), ""]

    # mat-mat
    mm = read_mm(os.path.join(d, "mm.csv"))
    if mm:
        md += ["## mat-mat (GFLOPS, tools/mm_bench)", ""]
        for gpu in sorted({r["gpu"] for r in mm}):
            for n in sorted({int(r["n"]) for r in mm if r["gpu"] == gpu}):
                rows = [r for r in mm if r["gpu"] == gpu and int(r["n"]) == n]
                cols = sorted({(r["type"], r["m"], r["k"]) for r in rows})
                variants = []
                for r in rows:
                    if r["variant"] not in variants:
                        variants.append(r["variant"])
                data = {(r["variant"], (r["type"], r["m"], r["k"])): r["gflops"] for r in rows}
                md += [f"### {gpu}, n={n}", "", table([[v] + [data.get((v, c), "-") for c in cols] for v in variants],
                                                     ["variant"] + [f"{t} m{m} k{k}" for t, m, k in cols]), ""]
                sp = speedups(mm, gpu, n)
                if sp:
                    ranked = sorted(sp.items(), key=lambda kv: -math.exp(sum(math.log(x) for x in kv[1]) / len(kv[1])))
                    md += ["geometric mean of the speed-up over `base` (quantized types): " +
                           ", ".join(f"{k} x{math.exp(sum(math.log(x) for x in v) / len(v)):.2f}" for k, v in ranked[:6]), ""]
    perf = read_mm(os.path.join(d, "perf.csv"))
    if perf:
        md += ["## test-backend-ops perf -o MUL_MAT, n=512 (GFLOPS)", ""]
        for gpu in sorted({r["gpu"] for r in perf}):
            rows = [r for r in perf if r["gpu"] == gpu]
            types = sorted({r["type"] for r in rows})
            variants = []
            for r in rows:
                if r["variant"] not in variants:
                    variants.append(r["variant"])
            data = {(r["variant"], r["type"]): r["gflops"] for r in rows}
            md += [f"### {gpu}", "", table([[v] + [data.get((v, t), "-") for t in types] for v in variants], ["variant"] + types), ""]

    # llama-bench
    bench = read_mm(os.path.join(d, "bench.csv"))
    if bench:
        md += ["## llama-bench (-ngl 99 -ub 128 -b 128, t/s)", ""]
        base = {(r["gpu"], r["model"]): r for r in bench if r["config"] == "nodn"}  # the dn stage: DN off is the reference
        base.update({(r["gpu"], r["model"]): r for r in bench if r["config"] == "base"})
        rows = []
        for r in bench:
            b = base.get((r["gpu"], r["model"]))
            def delta(key):
                try:
                    return f"{(float(r[key]) / float(b[key]) - 1) * 100:+.1f}%"
                except (TypeError, ValueError, ZeroDivisionError, KeyError):
                    return "-"
            rows.append([r["gpu"], r["model"], r["config"], r["pp128"], r["tg64"], delta("pp128") if b else "-", delta("tg64") if b else "-"])
        md += [table(rows, ["GPU", "model", "config", "pp128", "tg64", "pp vs base (nodn)", "tg vs base (nodn)"]), ""]

    # Fermi "denormal products": test-backend-ops / checks are in test.csv, the rest here
    dnmv = read_mm(os.path.join(d, "dn_mv.csv"))
    if dnmv:
        md += ["## Fermi denormal products: mat-vec (n=1, GB/s of the weights, tools/mm_bench; one launch of k_xprep per call)", ""]
        for gpu in sorted({r["gpu"] for r in dnmv}):
            rows = [r for r in dnmv if r["gpu"] == gpu]
            cols = sorted({(r["type"], r["m"], r["k"]) for r in rows})
            cfgs = []
            for r in rows:
                if r["config"] not in cfgs:
                    cfgs.append(r["config"])
            data = {(r["config"], (r["type"], r["m"], r["k"])): r["gbs"] for r in rows}
            body = [[c] + [data.get((c, col), "-") for col in cols] for c in cfgs]
            if "dn" in cfgs and "nodn" in cfgs:
                body.append(["dn / nodn"] + [f"x{float(data[('dn', col)]) / float(data[('nodn', col)]):.2f}" if ("dn", col) in data and ("nodn", col) in data and float(data[("nodn", col)]) > 0 else "-" for col in cols])
            md += [f"### {gpu}", "", table(body, ["config"] + [f"{t} m{m} k{k}" for t, m, k in cols]), ""]
    kl = read_mm(os.path.join(d, "kl.csv"))
    if kl:
        md += ["## KL divergence against the CPU (llama-perplexity -c 512 --chunks 8; tolerance KL_MEAN_MAX / KL_P99_MAX / KL_TOP_MIN = %s / %s / %s%%. The tolerance depends on the model and the text: calibrate it with an official backend on the same model and text, e.g. Vulkan on the RX 6400)" % (os.environ.get("KL_MEAN_MAX", "0.0070"), os.environ.get("KL_P99_MAX", "0.033"), os.environ.get("KL_TOP_MIN", "95.0")), "",
               table([[r["gpu"], r["model"], r["config"], r["ppl"], r["mean_kld"], r["kld99"], r["kld999"], r["max_dp"], r["same_top_p"], r["verdict"]] for r in kl],
                     ["GPU", "model", "config", "PPL", "mean KLD", "99% KLD", "99.9% KLD", "max dp %", "Same top p %", "result"]), ""]

    # regression of all the GPUs against the reference table (baseline.csv = README.md / docs/CL11.md)
    reg = read_mm(os.path.join(d, "regress.csv"))
    if reg:
        here = os.path.dirname(os.path.abspath(__file__))
        base = {r["gpu"]: r for r in read_mm(os.path.join(here, "baseline.csv"))}
        rows = []
        for r in reg:
            b = base.get(r["gpu"])
            tol = float(b["tol_pct"]) if b else 5.0

            def cell(key):
                v = r.get(key, "-")
                try:
                    fv = float(v)
                except ValueError:
                    return v
                if not b or key not in b:
                    return v
                rel = (fv / float(b[key]) - 1) * 100
                return f"{v} ({rel:+.1f}%{' REGRESSION' if rel < -tol else ''})"

            try:
                miss = int(r["test_total"]) - int(r["test_passed"])
                tres = "PASS" if int(r["test_total"]) > 0 and miss <= (int(b["allowed_fail"]) if b else 0) else "FAIL"
            except ValueError:
                tres = "-"
            rows.append([r["gpu"], f"{r['test_passed']}/{r['test_total']} {tres}", r["platform"], r["greedy_q8"], r["greedy_q4k"],
                         cell("q8_pp"), cell("q8_tg"), cell("q4k_pp"), cell("q4k_tg")])
        md += ["## regression of every GPU (default configuration; the value in brackets is the change against the reference table of README.md / docs/CL11.md, baseline.csv)", "",
               table(rows, ["GPU", "test-backend-ops", "platform", "greedy Q8_0 vs CPU", "greedy Q4_K_M vs main", "Q8_0 pp128", "Q8_0 tg64", "Q4_K_M pp128", "Q4_K_M tg64"]), ""]

    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(os.path.join(d, "summary.md"))


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return 2
    cmd = a[0]
    if cmd == "test":
        cmd_test(a[1])
    elif cmd == "perf":
        cmd_perf(a[1], a[2], a[3])
    elif cmd == "bench":
        cmd_bench(a[1])
    elif cmd == "kernels":
        cmd_kernels(a[1])
    elif cmd == "best":
        cmd_best(a[1], a[2], a[3], a[4])
    elif cmd == "mvgbs":
        cmd_mvgbs(a[1], a[2], a[3])
    elif cmd == "kl":
        cmd_kl(a[1])
    elif cmd == "summary":
        cmd_summary(a[1])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
