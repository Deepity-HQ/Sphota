"""Controls for the single-head PC attention prototype (attention_proto.py).

Same model (fixed random token codes + sinusoidal positions, one causal head,
linear readout over every position's context vector, window of past characters
-> next character), but with honest evaluation and a backprop reference:

  * Predictions are always made by a plain forward pass BEFORE the target is
    shown (free phase). The target only enters afterwards, to compute updates.
    attention_proto.py instead reads its prediction off a belief that has already
    been nudged by the target.
  * `--mode bp`  trains with exact backprop (manual numpy, finite-difference checked).
  * `--mode pc`  trains Q/K/V from the settled PC errors of Deepity's attention
    layers. Beliefs start at the free-phase values, so the errors contain only the
    nudge, and the readout error enters with the sign the layers expect
    (e_above = target - prediction, same as e = z - mu everywhere else).
    Wo/bo always use the exact cross-entropy gradient.

Subcommands:
    check     finite-difference check of the manual backprop
    compare   PC-vs-backprop gradient agreement (cosine, scale) over settling settings
    train     online next-character training with the chosen mode

Run from the Sphota repo root with the system python3 (the Deepity CPU extension
is ABI-tied to it, see attention_proto.py):
    PYTHONPATH=src python3 -m sphota.attention_checks check
    PYTHONPATH=src python3 -m sphota.attention_checks compare
    PYTHONPATH=src python3 -m sphota.attention_checks train --mode bp --tokens 100000
    PYTHONPATH=src python3 -m sphota.attention_checks train --mode pc --tokens 100000
"""

import argparse
import os
import sys

import numpy as np

from sphota.attention_proto import Adam, char_stream, sinusoidal_positional_encoding, softmax

_DEEPITY_BUILD_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "deepity", "build", "Release", "bin"
)


class Head:
    """One causal attention head + linear readout; forward and exact backward."""

    def __init__(self, vocab_size: int, embed_dim: int, head_dim: int, window: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.window, self.head_dim = window, head_dim
        self.emb = (rng.normal(size=(vocab_size, embed_dim)) / np.sqrt(embed_dim)).astype(np.float32)
        self.pos = sinusoidal_positional_encoding(window, embed_dim)
        self.W = {
            "q": (rng.normal(size=(embed_dim, head_dim)) / np.sqrt(embed_dim)).astype(np.float32),
            "k": (rng.normal(size=(embed_dim, head_dim)) / np.sqrt(embed_dim)).astype(np.float32),
            "v": (rng.normal(size=(embed_dim, head_dim)) / np.sqrt(embed_dim)).astype(np.float32),
            "o": (rng.normal(size=(window * head_dim, vocab_size)) / np.sqrt(window * head_dim)).astype(np.float32),
            "b": np.zeros(vocab_size, dtype=np.float32),
        }
        self.mask = np.triu(np.ones((window, window), dtype=bool), k=1)

    def forward(self, tokens: np.ndarray) -> dict[str, np.ndarray]:
        X = self.emb[tokens] + self.pos
        Q, K, V = X @ self.W["q"], X @ self.W["k"], X @ self.W["v"]
        S = (Q @ K.T) / np.sqrt(self.head_dim)
        S = np.where(self.mask, -1e9, S)
        A = softmax(S)
        C = A @ V
        p = softmax(C.ravel() @ self.W["o"] + self.W["b"])
        return {"X": X, "Q": Q, "K": K, "V": V, "A": A, "C": C, "p": p}

    def backward(self, f: dict[str, np.ndarray], target: int) -> dict[str, np.ndarray]:
        """Exact gradients of the cross-entropy loss w.r.t. Wq, Wk, Wv, Wo, bo."""
        dlog = f["p"].copy()
        dlog[target] -= 1.0
        dC = (self.W["o"] @ dlog).reshape(self.window, self.head_dim)
        dA = dC @ f["V"].T
        dV = f["A"].T @ dC
        dS = f["A"] * (dA - (dA * f["A"]).sum(axis=1, keepdims=True))   # softmax Jacobian
        dS[self.mask] = 0.0
        scale = 1.0 / np.sqrt(self.head_dim)
        dQ, dK = (dS @ f["K"]) * scale, (dS.T @ f["Q"]) * scale
        return {
            "q": f["X"].T @ dQ, "k": f["X"].T @ dK, "v": f["X"].T @ dV,
            "o": np.outer(f["C"].ravel(), dlog), "b": dlog,
        }


def pc_grads(head: Head, layers, f: dict[str, np.ndarray], target: int,
             steps: int, ir: float, nudge_sign: float = 1.0) -> dict[str, np.ndarray]:
    """Q/K/V gradients read off the settled errors of the attention layers.

    Beliefs start at the free-phase values (zero error), then the readout error enters the
    output layer. nudge_sign=+1 is the sign the layers expect (target - prediction);
    -1 reproduces what attention_proto.py passes.
    """
    scores, output = layers
    S, Hd = head.window, head.head_dim
    np.asarray(scores.beliefs())[0][:] = f["A"]
    np.asarray(output.beliefs())[0][:] = f["C"]

    scores.compute_mu_only(f["Q"].ravel(), f["K"].ravel())
    onehot = np.zeros_like(f["p"])
    onehot[target] = 1.0
    for _ in range(steps):
        scores.calculate_state(True)
        attnZ = np.asarray(scores.beliefs()).ravel()
        output.compute_mu_only(attnZ, f["V"].ravel())
        output.calculate_state(True)
        ctx = np.asarray(output.beliefs())[0]
        p = softmax(ctx.ravel() @ head.W["o"] + head.W["b"])
        eAbove = nudge_sign * (head.W["o"] @ (onehot - p)).reshape(S, Hd)
        output.update_state(ir, eAbove.ravel().astype(np.float32))
        scores.update_state(ir, np.asarray(output.errors()).ravel(), f["V"].ravel())

    scores.calculate_state(True)
    z = np.asarray(scores.beliefs())[0]
    output.compute_mu_only(z.ravel(), f["V"].ravel())
    output.calculate_state(True)
    eScores, eOutput = np.asarray(scores.errors())[0].copy(), np.asarray(output.errors())[0].copy()

    scale = 1.0 / np.sqrt(Hd)
    dmu = -eScores
    dmu[head.mask] = 0.0
    dQ, dK = (dmu @ f["K"]) * scale, (dmu.T @ f["Q"]) * scale
    dV = z.T @ (-eOutput)
    return {"q": f["X"].T @ dQ, "k": f["X"].T @ dK, "v": f["X"].T @ dV}


def make_layers(deepity_build: str, window: int, head_dim: int):
    sys.path.insert(0, deepity_build)
    import deepity as pd  # noqa: resolved at runtime from the sibling Deepity build
    return pd.AttentionScoresLayer(1, window, head_dim), pd.AttentionOutputLayer(1, window, head_dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


# ----------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------

def cmd_check(_args) -> None:
    """Finite-difference check of Head.backward on a tiny random problem (float64)."""
    head = Head(vocab_size=11, embed_dim=8, head_dim=4, window=5, seed=1)
    for k in head.W:
        head.W[k] = head.W[k].astype(np.float64)
    head.emb, head.pos = head.emb.astype(np.float64), head.pos.astype(np.float64)
    tokens, target = np.array([3, 1, 4, 1, 5]), 9

    def loss() -> float:
        return -np.log(head.forward(tokens)["p"][target])

    grads = head.backward(head.forward(tokens), target)
    rng, worst = np.random.default_rng(0), 0.0
    for name, w in head.W.items():
        for _ in range(12):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old, h = w[idx], 1e-6
            w[idx] = old + h
            up = loss()
            w[idx] = old - h
            down = loss()
            w[idx] = old
            fd = (up - down) / (2 * h)
            worst = max(worst, abs(fd - grads[name][idx]) / (abs(fd) + abs(grads[name][idx]) + 1e-12))
    print(f"[check] worst relative error of manual backprop vs finite differences: {worst:.2e}")
    print("[check] PASS" if worst < 1e-5 else "[check] FAIL")


def cmd_compare(args) -> None:
    """Cosine and scale of PC-derived Q/K/V gradients against exact backprop."""
    stream, vocab = char_stream(open(args.data, encoding="utf-8").read())
    head = Head(vocab, args.embed_dim, args.head_dim, args.window, seed=args.seed)
    layers = make_layers(args.deepity_build, args.window, args.head_dim)

    # A few hundred bp-trained steps first, so gradients are measured at a non-random point.
    if args.warmup:
        opts = {k: Adam(head.W[k], args.lr) for k in head.W}
        for i in range(args.window, args.window + args.warmup):
            f = head.forward(stream[i - args.window: i])
            for k, g in head.backward(f, int(stream[i])).items():
                opts[k].step(g)

    samples = [(stream[i - args.window: i], int(stream[i]))
               for i in range(100_000, 100_000 + args.samples)]
    print(f"[compare] {args.samples} samples, warmup {args.warmup} bp steps; "
          f"cosine to exact backprop (1 = same direction, -1 = ascent), ratio = |g_pc|/|g_bp|")
    print(f"{'sign':>5}{'steps':>7}{'ir':>7} | " + " | ".join(f"{k}: cos   ratio" for k in "qkv"))
    for sign in (1.0, -1.0):
        for steps, ir in [(5, 0.01), (5, 0.1), (20, 0.1), (50, 0.1)]:
            cs = {k: [] for k in "qkv"}
            rs = {k: [] for k in "qkv"}
            for tokens, target in samples:
                f = head.forward(tokens)
                bp, pc = head.backward(f, target), pc_grads(head, layers, f, target, steps, ir, sign)
                for k in "qkv":
                    cs[k].append(cosine(pc[k], bp[k]))
                    rs[k].append(np.linalg.norm(pc[k]) / (np.linalg.norm(bp[k]) + 1e-30))
            cells = " | ".join(f"{np.mean(cs[k]):+.3f} {np.median(rs[k]):7.4f}" for k in "qkv")
            print(f"{sign:>+5.0f}{steps:>7}{ir:>7} | {cells}")


def cmd_train(args) -> None:
    stream, vocab = char_stream(open(args.data, encoding="utf-8").read())
    stream = stream[: args.tokens] if args.tokens > 0 else stream
    head = Head(vocab, args.embed_dim, args.head_dim, args.window, seed=args.seed)
    opts = {k: Adam(head.W[k], args.lr) for k in head.W}
    layers = make_layers(args.deepity_build, args.window, args.head_dim) if args.mode == "pc" else None

    counts = np.bincount(stream[args.window:], minlength=vocab) / max(len(stream) - args.window, 1)
    unigram_bits = float(-(counts[counts > 0] * np.log2(counts[counts > 0])).sum())
    print(f"[reference] unigram entropy {unigram_bits:.3f} bits/char, most common char "
          f"{counts.max():.2%}, uniform {np.log2(vocab):.3f}")

    hits, bits, n = 0, 0.0, 0
    for i in range(args.window, len(stream)):
        tokens, target = stream[i - args.window: i], int(stream[i])
        f = head.forward(tokens)                              # free phase: prediction before the target is used
        bits -= np.log2(max(float(f["p"][target]), 1e-12))
        hits += int(np.argmax(f["p"]) == target)
        n += 1

        grads = head.backward(f, target)
        if args.mode == "pc":
            grads.update(pc_grads(head, layers, f, target, args.steps, args.ir, nudge_sign=1.0))
        for k, g in grads.items():
            if args.freeze_qk and k in ("q", "k"):
                continue                                      # attention routing stays at its random init
            opts[k].step(g)

        if n % args.log_every == 0:
            print(f"token {n} | hit {hits / n:.2%} | bits/char {bits / n:.3f}")
    print(f"\n[Result] mode={args.mode}{' freeze_qk' if args.freeze_qk else ''} lr={args.lr} {n} predictions | hit {hits / n:.2%} | bits/char {bits / n:.3f} "
          f"| unigram entropy {unigram_bits:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "compare", "train"])
    ap.add_argument("--data", default="tiny_shakespeare.txt")
    ap.add_argument("--deepity-build", default=_DEEPITY_BUILD_DEFAULT)
    ap.add_argument("--mode", choices=["bp", "pc"], default="bp")
    ap.add_argument("--tokens", type=int, default=100_000)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20, help="PC settling steps (pc mode)")
    ap.add_argument("--ir", type=float, default=0.1, help="PC inference rate (pc mode)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=2000, help="bp steps before measuring gradients (compare)")
    ap.add_argument("--samples", type=int, default=200, help="samples averaged by compare")
    ap.add_argument("--freeze-qk", action="store_true",
                    help="keep Wq/Wk at their random init while everything else trains")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25_000)
    args = ap.parse_args()
    {"check": cmd_check, "compare": cmd_compare, "train": cmd_train}[args.command](args)


if __name__ == "__main__":
    main()
