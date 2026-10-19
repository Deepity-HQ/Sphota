"""Tune on one stream segment, report on a later one, over several seeds.

Every run starts with an empty cache and a fresh PCN, so the tune segment
[0, tune_tokens) and the report segment [tune_tokens, tune_tokens + report_tokens)
share nothing. Hyperparameters are chosen by adaptive-gate bits/char on the tune
segment (seed 0) and are never looked at again on the report segment.

Search is staged to keep the run count small: cache (temp x eps) first at the default
gate rate, then gate_lr at the best cache setting.

Usage:
    python -m sphota.sweep                                   # 100k tune, 200k report, 3 seeds
    python -m sphota.sweep --tune-tokens 50000 --report-tokens 50000 --seeds 2
"""

import argparse
import gc
import itertools

import numpy as np
import torch

from sphota.main import SphotaLM, char_stream, load_text

DEFAULTS = {"cache_temp": 0.05, "cache_eps": 0.02, "gate_lr": 0.05}


def run(segment: np.ndarray, vocab_size: int, decode_map: dict[int, str],
        seed: int, **params) -> dict[str, float]:
    model = SphotaLM(vocab_size=vocab_size, cache_size=len(segment), seed=seed, **params)
    result = model.stream_learn(segment, decode_map, verbose=False).result()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def tune(segment, vocab_size, decode_map, temps, epss, gate_lrs) -> dict[str, float]:
    best, best_bpc = dict(DEFAULTS), float("inf")

    def try_config(cfg: dict[str, float]) -> None:
        nonlocal best, best_bpc
        r = run(segment, vocab_size, decode_map, seed=0, **cfg)
        print(f"  {cfg} -> adaptive {r['adaptive_bpc']:.3f} bpc, {r['adaptive_hit']:.2%} hit "
              f"(best fixed {r['best_fixed_bpc']:.3f})", flush=True)
        if r["adaptive_bpc"] < best_bpc:
            best, best_bpc = dict(cfg), r["adaptive_bpc"]

    print("[Tune] cache temp x eps")
    for temp, eps in itertools.product(temps, epss):
        try_config({**DEFAULTS, "cache_temp": temp, "cache_eps": eps})
    print("[Tune] gate lr")
    for lr in gate_lrs:
        if lr != best["gate_lr"]:
            try_config({**best, "gate_lr": lr})
    return best


def report(segment, vocab_size, decode_map, cfg: dict[str, float], seeds: int) -> None:
    results = [run(segment, vocab_size, decode_map, seed=s, **cfg) for s in range(seeds)]
    print(f"  config {cfg}")
    for key in ("cache_hit", "pcn_hit", "best_fixed_hit", "best_fixed_bpc", "route_bpc",
                "adaptive_hit", "adaptive_bpc", "ceiling"):
        vals = np.array([r[key] for r in results])
        fmt = ".3f" if key.endswith("bpc") else ".2%"
        print(f"  {key:<16}{format(vals.mean(), fmt):>9} +/- {format(vals.std(), fmt)}")
    gain = np.array([r["best_fixed_bpc"] - r["adaptive_bpc"] for r in results])
    print(f"  gate bpc gain over best fixed lam: {gain.mean():.3f} +/- {gain.std():.3f} "
          f"(positive in {int((gain > 0).sum())}/{seeds} seeds)")


def floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sphota tune/report sweep")
    ap.add_argument("--tune-tokens", type=int, default=100_000)
    ap.add_argument("--report-tokens", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temps", type=floats, default=[0.02, 0.05, 0.1])
    ap.add_argument("--epss", type=floats, default=[0.01, 0.02, 0.05])
    ap.add_argument("--gate-lrs", type=floats, default=[0.01, 0.05, 0.2])
    args = ap.parse_args()

    stream, vocab_size, decode_map = char_stream(load_text())
    t, r = args.tune_tokens, args.report_tokens
    tune_seg, report_seg = stream[:t], stream[t: t + r]
    print(f"[Split] tune tokens [0, {len(tune_seg)}), report tokens [{t}, {t + len(report_seg)})")

    best = tune(tune_seg, vocab_size, decode_map, args.temps, args.epss, args.gate_lrs)
    print(f"\n[Report] tuned config on held-out segment, {args.seeds} seeds")
    report(report_seg, vocab_size, decode_map, best, args.seeds)
    print("\n[Report] default config on the same segment, for comparison")
    report(report_seg, vocab_size, decode_map, DEFAULTS, args.seeds)
