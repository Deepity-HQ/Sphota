"""CPU-only bench for the associative cache, no PCN and no GPU.

Streams a corpus through one cache backend and reports what the cache alone can do
(token hit rate and smoothed bits/char) plus what it costs (time per token against
stream position, candidates examined, memory, and for approximate backends the recall
of the exact top-k). The key, the vote weighting and the smoothing follow FastCache and
SphotaLM in main.py, so the exact backend is the reference for the others.

Backends:
    exact   brute-force cosine over every stored key (O(N) per token; the current cache)
    ring    the same search over a fixed-capacity ring (O(cap), forgets the oldest keys)
    lsh     random-hyperplane hash tables with fixed-size buckets over a fixed-capacity
            ring: O(1) time and space per token by construction
    ngram   hash of the last n tokens -> fixed-size ring of recent next tokens, backing
            off to shorter contexts (not a cosine cache; the strong simple baseline)

Usage:
    python -m sphota.cache_bench --backend exact --tokens 100000
    python -m sphota.cache_bench --backend lsh --cap 100000 --bits 12 --tables 8
    python -m sphota.cache_bench --suite
    python -m sphota.cache_bench --pair --cap 300000 --bits 14 --n 4 --tokens 300000
"""

import argparse
import time

import numpy as np

EMBED_DIM = 128
TRACE_DECAYS = (0.5, 0.9)
KEY_DIM = EMBED_DIM * (1 + len(TRACE_DECAYS))


# ----------------------------------------------------------------------------
# Voting
# ----------------------------------------------------------------------------

def topk_votes(next_tokens: np.ndarray, scores: np.ndarray, k: int, temp: float, vocab: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Similarity-weighted vote of the top-k candidates. Returns (distribution, chosen indices)."""
    n = len(scores)
    kk = min(k, n)
    idx = np.argpartition(-scores, kk - 1)[:kk] if n > kk else np.arange(n)
    idx = idx[np.argsort(-scores[idx])]
    s = scores[idx]
    w = np.exp((s - s[0]) / temp) if temp > 0 else np.ones(kk)
    votes = np.bincount(next_tokens[idx], weights=w, minlength=vocab)
    return votes / votes.sum(), idx


# ----------------------------------------------------------------------------
# Backends. vote() sees only the normalised key; add() stores it with the true next token.
# ----------------------------------------------------------------------------

class Ring:
    """Fixed-capacity store of normalised keys + next tokens, searched exhaustively.

    capacity >= stream length gives the unbounded exact cache.
    """

    def __init__(self, capacity: int, vocab: int, k: int, temp: float):
        self.cap, self.vocab, self.k, self.temp = capacity, vocab, k, temp
        self.keys = np.zeros((capacity, KEY_DIM), dtype=np.float32)
        self.toks = np.zeros(capacity, dtype=np.int64)
        self.count = 0
        self.last_gids: np.ndarray | None = None

    def _gid(self, slots: np.ndarray) -> np.ndarray:
        """Global insertion index currently held in each slot."""
        return (self.count - 1) - ((self.count - 1 - slots) % self.cap)

    def _search(self, q: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        m = min(self.count, self.cap)
        if m == 0:
            return None, None
        scores = self.keys[:m] @ q
        votes, idx = topk_votes(self.toks[:m], scores, self.k, self.temp, self.vocab)
        return votes, self._gid(idx)

    def vote(self, q: np.ndarray, i: int) -> tuple[np.ndarray | None, int]:
        votes, self.last_gids = self._search(q)
        return votes, min(self.count, self.cap)

    def add(self, q: np.ndarray, tok: int, i: int) -> None:
        slot = self.count % self.cap
        self.keys[slot], self.toks[slot] = q, tok
        self.count += 1

    def probe(self, q: np.ndarray) -> float | None:
        return None                                   # already exact

    def memory_bytes(self) -> int:
        return self.keys.nbytes + self.toks.nbytes


class Lsh(Ring):
    """Random-hyperplane LSH: `tables` tables of 2**bits buckets, each bucket a ring of the
    `bucket` newest item ids. A query reads one bucket per table and scores only those items."""

    def __init__(self, capacity: int, vocab: int, k: int, temp: float,
                 bits: int, tables: int, bucket: int, seed: int = 0):
        super().__init__(capacity, vocab, k, temp)
        self.bits, self.tables, self.bucket = bits, tables, bucket
        rng = np.random.default_rng(seed + 1)
        self.proj = rng.standard_normal((tables * bits, KEY_DIM)).astype(np.float32)
        self.pow2 = (1 << np.arange(bits)).astype(np.int64)
        self.ids = np.full((tables, 1 << bits, bucket), -1, dtype=np.int64)
        self.ptr = np.zeros((tables, 1 << bits), dtype=np.int64)
        self.rows = np.arange(tables)
        self._hs: np.ndarray | None = None

    def _hash(self, q: np.ndarray) -> np.ndarray:
        return (self.proj @ q > 0).reshape(self.tables, self.bits).astype(np.int64) @ self.pow2

    def vote(self, q: np.ndarray, i: int) -> tuple[np.ndarray | None, int]:
        self._hs = self._hash(q)
        cand = self.ids[self.rows, self._hs].ravel()
        cand = np.unique(cand[cand >= max(0, self.count - self.cap)])    # drops -1 and evicted items
        if len(cand) == 0:
            self.last_gids = None
            return None, 0
        slots = cand % self.cap
        votes, idx = topk_votes(self.toks[slots], self.keys[slots] @ q, self.k, self.temp, self.vocab)
        self.last_gids = cand[idx]
        return votes, len(cand)

    def add(self, q: np.ndarray, tok: int, i: int) -> None:
        gid = self.count
        super().add(q, tok, i)
        slot_in_bucket = self.ptr[self.rows, self._hs] % self.bucket
        self.ids[self.rows, self._hs, slot_in_bucket] = gid
        self.ptr[self.rows, self._hs] += 1

    def probe(self, q: np.ndarray) -> float | None:
        """Overlap of this query's approximate top-k with the exact top-k over the live ring."""
        if self.last_gids is None or self.count == 0:
            return 0.0
        _, exact = Ring._search(self, q)
        if exact is None:
            return 0.0
        return len(np.intersect1d(self.last_gids, exact)) / len(exact)

    def memory_bytes(self) -> int:
        return super().memory_bytes() + self.ids.nbytes + self.ptr.nbytes + self.proj.nbytes


class NgramHash:
    """Hash of the last o tokens (o = n .. 1) -> ring of recent next tokens; votes by count,
    using the longest context that has been seen."""

    def __init__(self, n: int, vocab: int, stream: np.ndarray, buckets: int = 1 << 20, bucket: int = 16):
        self.n, self.vocab, self.stream, self.nb, self.bucket = n, vocab, stream, buckets, bucket
        self.toks = [np.full((buckets, bucket), -1, dtype=np.int16) for _ in range(n)]
        self.ptr = [np.zeros(buckets, dtype=np.int64) for _ in range(n)]
        self.last_gids = None

    def _bucket(self, i: int, order: int) -> int:
        h = 0
        for t in self.stream[i - order: i]:
            h = (h * 1000003 + int(t) + 1) % self.nb
        return h

    def vote(self, q: np.ndarray, i: int) -> tuple[np.ndarray | None, int]:
        for order in range(min(self.n, i), 0, -1):
            row = self.toks[order - 1][self._bucket(i, order)]
            seen = row[row >= 0]
            if len(seen):
                votes = np.bincount(seen, minlength=self.vocab).astype(np.float64)
                return votes / votes.sum(), len(seen)
        return None, 0

    def add(self, q: np.ndarray, tok: int, i: int) -> None:
        for order in range(1, min(self.n, i) + 1):
            b = self._bucket(i, order)
            self.toks[order - 1][b, self.ptr[order - 1][b] % self.bucket] = tok
            self.ptr[order - 1][b] += 1

    def probe(self, q: np.ndarray) -> float | None:
        return None

    def memory_bytes(self) -> int:
        return sum(t.nbytes for t in self.toks) + sum(p.nbytes for p in self.ptr)


class InterpNgram:
    """Witten-Bell interpolated n-gram over orders 0..n with exact counts.

    A proper smoothed baseline for the question "does the cosine cache add anything beyond a
    good n-gram?". Memory is unbounded; this is a quality control, not an O(1) structure.
    p_o(c) = (C(h,c) + T(h) * p_{o-1}(c)) / (C(h) + T(h)), T(h) = number of distinct next tokens.
    """

    def __init__(self, n: int, vocab: int, stream: np.ndarray):
        self.n, self.vocab, self.stream = n, vocab, stream
        self.counts: list[dict[bytes, np.ndarray]] = [{} for _ in range(n + 1)]
        self.last_gids = None
        self.last_order, self.last_total = 0, 0.0        # deepest context used by the last vote

    def _ctx(self, i: int, order: int) -> bytes:
        return self.stream[i - order: i].tobytes() if order else b""

    def vote(self, q: np.ndarray, i: int) -> tuple[np.ndarray | None, int]:
        p = np.full(self.vocab, 1.0 / self.vocab)
        self.last_order, self.last_total = 0, 0.0
        for order in range(min(self.n, i) + 1):
            c = self.counts[order].get(self._ctx(i, order))
            if c is None:
                break                                  # a longer context cannot have been seen either
            total, distinct = float(c.sum()), float(np.count_nonzero(c))
            lam = total / (total + distinct)
            p = lam * c / total + (1.0 - lam) * p
            self.last_order, self.last_total = order, total
        return p, 1

    def add(self, q: np.ndarray, tok: int, i: int) -> None:
        for order in range(min(self.n, i) + 1):
            key = self._ctx(i, order)
            c = self.counts[order].get(key)
            if c is None:
                c = self.counts[order][key] = np.zeros(self.vocab, dtype=np.float32)
            c[tok] += 1

    def probe(self, q: np.ndarray) -> float | None:
        return None

    def memory_bytes(self) -> int:
        return sum(len(d) * (self.vocab * 4 + 100) for d in self.counts)


class HashedInterpNgram:
    """Witten-Bell interpolation over orders 0..n in fixed-size tables: O(1) time and space.

    Order o keeps `rows` count vectors. When vocab**o <= rows the context indexes its row
    directly (exact); otherwise it is hashed and each row carries a tag of a second hash. A
    tag mismatch reads as "unseen", and a write to a mismatching row evicts it (newest wins),
    so collisions never mix two contexts' counts.
    """

    def __init__(self, n: int, vocab: int, stream: np.ndarray, rows_bits: int = 18):
        self.n, self.vocab, self.stream = n, vocab, stream
        rows = 1 << rows_bits
        self.sizes = [min(rows, vocab ** o) for o in range(n + 1)]
        self.exact = [vocab ** o <= rows for o in range(n + 1)]
        self.counts = [np.zeros((sz, vocab), dtype=np.float32) for sz in self.sizes]
        self.tags = [None if ex else np.zeros(sz, dtype=np.uint32) for sz, ex in zip(self.sizes, self.exact)]
        self.last_gids = None
        self.last_order, self.last_total = 0, 0.0        # deepest context used by the last vote

    def _locate(self, i: int, order: int) -> tuple[int, int]:
        ctx = self.stream[i - order: i]
        if self.exact[order]:
            idx = 0
            for t in ctx:
                idx = idx * self.vocab + int(t)
            return idx, 0
        h1, h2 = 0, 0
        for t in ctx:
            h1 = (h1 * 1000003 + int(t) + 1) & 0xFFFFFFFFFFFF
            h2 = (h2 * 2654435761 + int(t) + 7) & 0xFFFFFFFF
        return h1 % self.sizes[order], (h2 | 1) & 0xFFFFFFFF

    def vote(self, q: np.ndarray, i: int) -> tuple[np.ndarray | None, int]:
        p = np.full(self.vocab, 1.0 / self.vocab)
        self.last_order, self.last_total = 0, 0.0
        for order in range(min(self.n, i) + 1):
            idx, tag = self._locate(i, order)
            if not self.exact[order] and self.tags[order][idx] != tag:
                break
            c = self.counts[order][idx]
            total = float(c.sum())
            if total == 0.0:
                break
            lam = total / (total + float(np.count_nonzero(c)))
            p = lam * c / total + (1.0 - lam) * p
            self.last_order, self.last_total = order, total
        return p, 1

    def add(self, q: np.ndarray, tok: int, i: int) -> None:
        for order in range(min(self.n, i) + 1):
            idx, tag = self._locate(i, order)
            if not self.exact[order] and self.tags[order][idx] != tag:
                self.counts[order][idx] = 0.0
                self.tags[order][idx] = tag
            self.counts[order][idx, tok] += 1

    def probe(self, q: np.ndarray) -> float | None:
        return None

    def memory_bytes(self) -> int:
        return sum(c.nbytes for c in self.counts) + sum(t.nbytes for t in self.tags if t is not None)


# ----------------------------------------------------------------------------
# Stream + metrics
# ----------------------------------------------------------------------------

def load_stream(path: str) -> tuple[np.ndarray, int]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    index = {c: j for j, c in enumerate(chars)}
    return np.array([index[c] for c in text], dtype=np.int32), len(chars)


def run(backend, stream: np.ndarray, vocab: int, eps: float, seed: int = 0, log_every: int = 0,
        probe_every: int = 500, label: str = "") -> dict[str, float]:
    emb = np.random.default_rng(seed).standard_normal((vocab, EMBED_DIM)).astype(np.float32)
    traces = [np.zeros(EMBED_DIM, dtype=np.float32) for _ in TRACE_DECAYS]
    hits, bits, cands, empty, n = 0, 0.0, 0, 0, 0
    recalls: list[float] = []
    block_time, block_n, block_ms = 0.0, 0, []
    checkpoints: dict[int, tuple[float, float]] = {}

    for i in range(1, len(stream)):
        prev, tok = int(stream[i - 1]), int(stream[i])
        ctx = np.concatenate([emb[prev], *traces])
        q = (ctx / (np.linalg.norm(ctx) + 1e-12)).astype(np.float32)

        t0 = time.perf_counter()
        votes, ncand = backend.vote(q, i)
        dt = time.perf_counter() - t0
        if probe_every and i % probe_every == 0:
            r = backend.probe(q)
            if r is not None:
                recalls.append(r)
        t0 = time.perf_counter()
        backend.add(q, tok, i)
        block_time += dt + time.perf_counter() - t0

        n += 1
        cands += ncand
        if votes is None:
            empty += 1
            bits -= np.log2(1.0 / vocab)
        else:
            hits += int(np.argmax(votes) == tok)
            bits -= np.log2((1.0 - eps) * votes[tok] + eps / vocab)

        traces = [d * t + (1.0 - d) * emb[prev] for d, t in zip(TRACE_DECAYS, traces)]

        block_n += 1
        if block_n == (log_every or 50_000):
            block_ms.append(1000.0 * block_time / block_n)
            block_time, block_n = 0.0, 0
            checkpoints[n] = (hits / n, bits / n)
            if log_every:
                print(f"  {label} token {n} | hit {hits / n:.2%} | bits/char {bits / n:.3f} | "
                      f"{block_ms[-1]:.3f} ms/token | cands {cands / n:.0f}", flush=True)

    return {
        "hit": hits / n, "bpc": bits / n, "empty": empty / n, "cands": cands / n,
        "recall": float(np.mean(recalls)) if recalls else float("nan"),
        "ms_first": block_ms[0] if block_ms else float("nan"),
        "ms_last": block_ms[-1] if block_ms else float("nan"),
        "mem_mb": backend.memory_bytes() / 1e6,
    }


PAIR_LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run_pair(cache, ngram, stream: np.ndarray, vocab: int, eps: float, gate_lr: float = 0.05,
             seed: int = 0, log_every: int = 0, eps_b: float | None = None) -> dict[str, float]:
    """Cosine cache (a) and hashed n-gram (b) on one stream: who is right when, and what a mix buys.

    p_mix = lam * p_ngram + (1 - lam) * p_cache, both smoothed like the single-backend runs.
    `gate` learns lam online from vote confidence by SGD on the mixture log-loss (as in main.py).
    """
    emb = np.random.default_rng(seed).standard_normal((vocab, EMBED_DIM)).astype(np.float32)
    traces = [np.zeros(EMBED_DIM, dtype=np.float32) for _ in TRACE_DECAYS]
    uniform = np.full(vocab, 1.0 / vocab)
    names = ["cache", "ngram"] + [f"mix{lam}" for lam in PAIR_LAMS[1:-1]] + ["gate"]
    hits = dict.fromkeys(names, 0)
    bits = dict.fromkeys(names, 0.0)
    agree = {"both": 0, "cache_only": 0, "ngram_only": 0, "neither": 0}
    lam_sum = {k: 0.0 for k in agree}
    gate_w = np.zeros(5)
    n = 0

    def entropy(p: np.ndarray) -> float:
        return float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(vocab))

    for i in range(1, len(stream)):
        prev, tok = int(stream[i - 1]), int(stream[i])
        ctx = np.concatenate([emb[prev], *traces])
        q = (ctx / (np.linalg.norm(ctx) + 1e-12)).astype(np.float32)

        va, _ = cache.vote(q, i)
        vb, _ = ngram.vote(q, i)
        va_ = uniform if va is None else va
        vb_ = uniform if vb is None else vb
        pa = (1.0 - eps) * va_ + eps / vocab
        eb = eps if eps_b is None else eps_b
        pb = (1.0 - eb) * vb_ + eb / vocab

        feats = np.array([va_.max(), entropy(va_), vb_.max(), entropy(vb_), 1.0])
        lam_gate = 1.0 / (1.0 + np.exp(-gate_w @ feats))
        mixes = {"cache": pa, "ngram": pb, **{f"mix{lam}": lam * pb + (1 - lam) * pa for lam in PAIR_LAMS[1:-1]},
                 "gate": lam_gate * pb + (1 - lam_gate) * pa}
        for name, p in mixes.items():
            hits[name] += int(np.argmax(p) == tok)
            bits[name] -= np.log2(max(p[tok], 1e-12))

        a_ok, b_ok = int(np.argmax(pa)) == tok, int(np.argmax(pb)) == tok
        key = "both" if a_ok and b_ok else "cache_only" if a_ok else "ngram_only" if b_ok else "neither"
        agree[key] += 1
        lam_sum[key] += lam_gate

        pm = max(lam_gate * pb[tok] + (1 - lam_gate) * pa[tok], 1e-12)
        gate_w -= gate_lr * (-(pb[tok] - pa[tok]) / pm * lam_gate * (1 - lam_gate)) * feats

        cache.add(q, tok, i)
        ngram.add(q, tok, i)
        traces = [d * t + (1.0 - d) * emb[prev] for d, t in zip(TRACE_DECAYS, traces)]
        n += 1
        if log_every and n % log_every == 0:
            print(f"  token {n} | cache {hits['cache'] / n:.2%} | ngram {hits['ngram'] / n:.2%} | "
                  f"gate {hits['gate'] / n:.2%} {bits['gate'] / n:.3f} bits", flush=True)

    out = {f"{k}_hit": hits[k] / n for k in names} | {f"{k}_bpc": bits[k] / n for k in names}
    out |= {f"agree_{k}": v / n for k, v in agree.items()}
    out |= {f"lam_{k}": lam_sum[k] / max(agree[k], 1) for k in agree}
    out["ceiling"] = 1.0 - agree["neither"] / n
    return out


def print_pair(r: dict[str, float]) -> None:
    print(f"{'predictor':<12}{'hit':>8}{'bits/char':>11}")
    for name in ["cache", "ngram", "mix0.25", "mix0.5", "mix0.75", "gate"]:
        print(f"{name:<12}{r[name + '_hit']:>8.2%}{r[name + '_bpc']:>11.3f}")
    print("cache vs ngram: " + ", ".join(f"{k} {r['agree_' + k]:.2%}" for k in ("both", "cache_only", "ngram_only", "neither")))
    print("gate mean lam (weight on ngram): " + " | ".join(f"{k} {r['lam_' + k]:.3f}" for k in ("both", "cache_only", "ngram_only", "neither")))
    print(f"ceiling (either right): {r['ceiling']:.2%}")


def make_backend(args, stream: np.ndarray, vocab: int, cap: int | None = None):
    cap = cap or args.cap or len(stream)
    if args.backend == "exact":
        return Ring(len(stream), vocab, args.top_k, args.temp)
    if args.backend == "ring":
        return Ring(cap, vocab, args.top_k, args.temp)
    if args.backend == "lsh":
        return Lsh(cap, vocab, args.top_k, args.temp, args.bits, args.tables, args.bucket)
    if args.backend == "interp":
        return InterpNgram(args.n, vocab, stream)
    if args.backend == "hinterp":
        return HashedInterpNgram(args.n, vocab, stream, args.rows_bits)
    return NgramHash(args.n, vocab, stream)


SUITE = [
    ("exact (reference)", dict(backend="exact"), "exact"),
    ("ring 20k", dict(backend="ring", cap=20_000), "main"),
    ("ring 50k", dict(backend="ring", cap=50_000), "main"),
    ("lsh 12b x8, cap 100k", dict(backend="lsh", cap=100_000, bits=12, tables=8, bucket=32), "main"),
    ("lsh 12b x8, cap 300k", dict(backend="lsh", cap=300_000, bits=12, tables=8, bucket=32), "main"),
    ("lsh 14b x8, cap 300k", dict(backend="lsh", cap=300_000, bits=14, tables=8, bucket=32), "main"),
    ("ngram n=2", dict(backend="ngram", n=2), "main"),
    ("ngram n=3", dict(backend="ngram", n=3), "main"),
    ("ngram n=4", dict(backend="ngram", n=4), "main"),
]


def print_row(label: str, r: dict[str, float]) -> None:
    print(f"{label:<24}{r['hit']:>8.2%}{r['bpc']:>8.3f}{r['empty']:>8.2%}{r['cands']:>9.0f}"
          f"{r['recall']:>8.2f}{r['ms_first']:>9.3f}{r['ms_last']:>9.3f}{r['mem_mb']:>9.0f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="tiny_shakespeare.txt")
    ap.add_argument("--backend", choices=["exact", "ring", "lsh", "ngram", "interp", "hinterp"], default="exact")
    ap.add_argument("--tokens", type=int, default=0, help="stream length (0 = whole corpus)")
    ap.add_argument("--exact-tokens", type=int, default=100_000, help="suite: stream length for the O(N) reference")
    ap.add_argument("--cap", type=int, default=0, help="ring/lsh capacity (0 = stream length)")
    ap.add_argument("--bits", type=int, default=12)
    ap.add_argument("--tables", type=int, default=8)
    ap.add_argument("--bucket", type=int, default=32)
    ap.add_argument("--n", type=int, default=3, help="ngram order")
    ap.add_argument("--rows-bits", type=int, default=18, help="hinterp: log2 rows per order's count table")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--suite", action="store_true", help="run the standard comparison and print one table")
    ap.add_argument("--pair", action="store_true", help="run the LSH cache and an n-gram together")
    ap.add_argument("--pair-ngram", choices=["hash", "interp"], default="hash",
                    help="pair: the n-gram partner (hash = O(1) bucketed counts, interp = Witten-Bell control)")
    ap.add_argument("--only", default="", help="suite: run only the rows whose label contains this text")
    args = ap.parse_args()

    stream, vocab = load_stream(args.data)
    header = (f"{'backend':<24}{'hit':>8}{'bits/ch':>8}{'empty':>8}{'cands':>9}{'recall':>8}"
              f"{'ms first':>9}{'ms last':>9}{'MB':>9}")

    if args.suite:
        full = stream[: args.tokens] if args.tokens else stream
        print(f"[suite] {len(full)} tokens ({args.exact_tokens} for the exact reference); "
              f"ms first/last = time per token in the first/last 50k-token block\n{header}")
        for label, over, which in SUITE:
            if args.only and args.only not in label:
                continue
            a = argparse.Namespace(**{**vars(args), **over})
            seg = stream[: args.exact_tokens] if which == "exact" else full
            print_row(label, run(make_backend(a, seg, vocab), seg, vocab, args.eps, args.seed))
        return

    seg = stream[: args.tokens] if args.tokens else stream
    if args.pair:
        cache = Lsh(args.cap or len(seg), vocab, args.top_k, args.temp, args.bits, args.tables, args.bucket)
        print(f"[pair] {len(seg)} tokens | lsh cap {cache.cap} {args.bits}b x{args.tables} bucket {args.bucket} "
              f"| ngram={args.pair_ngram} n={args.n}")
        partner = InterpNgram(args.n, vocab, seg) if args.pair_ngram == "interp" else NgramHash(args.n, vocab, seg)
        print_pair(run_pair(cache, partner, seg, vocab, args.eps, seed=args.seed, log_every=args.log_every))
        return
    print(header)
    print_row(args.backend, run(make_backend(args, seg, vocab), seg, vocab, args.eps, args.seed,
                                args.log_every, label=args.backend))


if __name__ == "__main__":
    main()
