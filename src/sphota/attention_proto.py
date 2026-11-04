"""Standalone proof of concept: does Deepity's PC attention mechanism
(AttentionScoresLayer + AttentionOutputLayer, settled via the categorical/
KL energy from Pinchetti et al. arXiv:2211.03481) learn anything real on
actual text, with Q/K/V/readout weights genuinely updated from feedback
flowing back through the attention block?

Deliberately standalone, not wired into SphotaLM (see main.py) yet:
  * CPU-only -- no batched GPU GEMM exists for the new C++ classes, and
    this isn't the throughput-tuned live pipeline.
  * Doesn't depend on pydeepity (not built in this checkout, and ABI-tied
    to a different Python minor version than this venv) -- only needs
    the plain `deepity` CPU extension from the sibling Deepity checkout,
    plus numpy. Q/K/V projections and the readout are hand-rolled numpy,
    not FullPCLayer: FullPCLayer's own settling machinery expects its
    "layer above" to literally be another FullPCLayer, which the
    attention block isn't (see AttentionScoresLayer.h's doc comment).
    Wiring real FullPCLayer projections is the next, separate piece.
  * One sequence window at a time (batch=1), predicting the next token
    from the window's last position -- proving the mechanism on real
    data before any decision about live-pipeline integration.

Usage (run from the Sphota repo root with the plain system python3, not
the project's uv venv -- see the docstring above for why):
    python3 -m sphota.attention_proto --tokens 20000
"""

import argparse
import os
import sys

import numpy as np

_DEEPITY_BUILD_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "deepity", "build", "Release", "bin"
)


class Adam:
    """Per-parameter Adam moments, applied in place to `value` (a numpy
    array owned by the caller, updated in place via -=)."""

    def __init__(self, value: np.ndarray, lr: float, beta1: float = 0.9,
                beta2: float = 0.999, eps: float = 1e-8):
        self.value, self.lr = value, lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = np.zeros_like(value, dtype=np.float64)
        self.v = np.zeros_like(value, dtype=np.float64)
        self.t = 0

    def step(self, grad: np.ndarray) -> None:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        mHat = self.m / (1 - self.beta1 ** self.t)
        vHat = self.v / (1 - self.beta2 ** self.t)
        self.value -= (self.lr * mHat / (np.sqrt(vHat) + self.eps)).astype(self.value.dtype)


def sinusoidal_positional_encoding(window: int, embed_dim: int) -> np.ndarray:
    """Standard fixed (non-learned) positional encoding (Vaswani et al.,
    "Attention is All You Need"), shape [window, embed_dim]. Added
    directly to token embeddings, needs no gradient of its own -- the
    cheapest way to test whether missing positional information (not
    capacity or readout width, both already ruled out) is the reason
    this model can't beat the unigram floor: Q/K are otherwise computed
    purely from each token's own embedding, so two windows with the same
    characters in a different order look identical to the attention
    mechanism except for the causal mask itself.
    """
    pos = np.arange(window)[:, None]
    i = np.arange(embed_dim)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / embed_dim)
    pe = np.zeros((window, embed_dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def clip_norm(grad: np.ndarray, max_norm: float) -> np.ndarray:
    """Rescales grad to have L2 norm at most max_norm (no-op if already smaller).

    Defends an explicit-Euler PC settling loop (no stability guarantee of
    its own) and Adam's adaptive per-parameter step size against the rare
    input where settling overshoots badly enough to produce a huge
    gradient -- confirmed to happen in practice (NaN after ~25-60k tokens
    without this), not a hypothetical.
    """
    norm = np.linalg.norm(grad)
    return grad * (max_norm / norm) if norm > max_norm else grad


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def char_stream(text: str) -> tuple[np.ndarray, int]:
    chars = sorted(set(text))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    stream = np.array([char_to_int[c] for c in text], dtype=np.int32)
    print(f"Loaded text (chars): {len(stream)} tokens, {len(chars)} vocab size")
    return stream, len(chars)


class AttentionProto:
    """One causal attention head over a sliding window, settled via PC,
    predicting the next token from the whole settled window (every
    position's context vector feeds the readout, not just the last).

    Q/K/V and the readout are plain numpy linear projections, updated
    each step from the settled attention block's own error buffers --
    e = z - softmax(mu) is exactly d(categorical CE)/dmu, and e = z - mu
    is exactly d(Gaussian energy)/dmu (both already established facts
    about these kernels, not assumptions), so no separate backprop
    machinery is needed to turn the settled errors into weight
    gradients.
    """

    def __init__(self, pd, vocab_size: int, embed_dim: int = 64, head_dim: int = 32,
                window: int = 16, settle_steps: int = 5, inference_rate: float = 0.01,
                lr: float = 1e-2, grad_clip: float = 5.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab_size, self.head_dim, self.window = vocab_size, head_dim, window
        self.settle_steps, self.ir, self.lr = settle_steps, inference_rate, lr
        self.grad_clip = grad_clip

        readout_dim = window * head_dim
        self.emb = (rng.normal(size=(vocab_size, embed_dim)) / np.sqrt(embed_dim)).astype(np.float32)
        self.Wq = (rng.normal(size=(embed_dim, head_dim)) / np.sqrt(embed_dim)).astype(np.float32)
        self.Wk = (rng.normal(size=(embed_dim, head_dim)) / np.sqrt(embed_dim)).astype(np.float32)
        self.Wv = (rng.normal(size=(embed_dim, head_dim)) / np.sqrt(embed_dim)).astype(np.float32)
        self.Wo = (rng.normal(size=(readout_dim, vocab_size)) / np.sqrt(readout_dim)).astype(np.float32)
        self.bo = np.zeros(vocab_size, dtype=np.float32)
        self.posEnc = sinusoidal_positional_encoding(window, embed_dim)

        # Zero gradient through a causally-masked entry: mu there is the
        # constant kMaskValue, not a function of Q/K, so d(mu)/d(Q,K) is
        # really 0 there even though the settled z/e are only near-0.
        self.causalMask = np.triu(np.ones((window, window), dtype=bool), k=1)

        self.scores = pd.AttentionScoresLayer(1, window, head_dim)
        self.output = pd.AttentionOutputLayer(1, window, head_dim)

        self.optWq = Adam(self.Wq, lr)
        self.optWk = Adam(self.Wk, lr)
        self.optWv = Adam(self.Wv, lr)
        self.optWo = Adam(self.Wo, lr)
        self.optBo = Adam(self.bo, lr)

    def step(self, window_tokens: np.ndarray, target_token: int, learn: bool = True
             ) -> tuple[float, int]:
        # Reset both layers' settled belief before each token: without
        # this they warm-start from the previous token's final z, and
        # with the output layer's unbounded Gaussian energy that lets
        # drift accumulate across many tokens until it overflows --
        # confirmed in practice (gradual NaN after ~25-60k tokens), not
        # hypothetical. Each one's own settling behavior was validated
        # starting from z=0, never from a carried-over state.
        np.asarray(self.scores.beliefs())[:] = 0.0
        np.asarray(self.output.beliefs())[:] = 0.0

        X = self.emb[window_tokens] + self.posEnc  # [S, D]
        Q = (X @ self.Wq).astype(np.float32)
        K = (X @ self.Wk).astype(np.float32)
        V = (X @ self.Wv).astype(np.float32)

        for _ in range(self.settle_steps):
            self.scores.compute_mu_only(Q.ravel(), K.ravel())
            self.scores.calculate_state(True)
            attnZ = np.asarray(self.scores.beliefs()).ravel()
            self.output.compute_mu_only(attnZ, V.ravel())
            self.output.calculate_state(True)

            ctxAll = np.asarray(self.output.beliefs())[0]  # [S, Hd], every position
            p = softmax(ctxAll.ravel() @ self.Wo + self.bo)
            dlogits = p.copy()
            dlogits[target_token] -= 1.0
            eAboveOutput = (self.Wo @ dlogits).reshape(self.window, self.head_dim)

            self.output.update_state(self.ir, eAboveOutput.ravel())
            eOutputFlat = np.asarray(self.output.errors()).ravel()
            self.scores.update_state(self.ir, eOutputFlat, V.ravel())

        z = np.asarray(self.scores.beliefs())[0]      # [S, S], settled attention weights
        eScores = np.asarray(self.scores.errors())[0]  # [S, S] = z - softmax(mu)
        eOutput = np.asarray(self.output.errors())[0]  # [S, Hd] = ctxZ - mu_output
        ctxAll = np.asarray(self.output.beliefs())[0]
        logits = ctxAll.ravel() @ self.Wo + self.bo
        p = softmax(logits)
        loss_bits = -np.log2(max(float(p[target_token]), 1e-12))
        pred = int(np.argmax(p))

        if learn:
            dlogits = p.copy()
            dlogits[target_token] -= 1.0
            dWo, dbo = np.outer(ctxAll.ravel(), dlogits), dlogits

            scale = 1.0 / np.sqrt(self.head_dim)
            dmuScores = -eScores
            dmuScores[self.causalMask] = 0.0
            dmuOutput = -eOutput

            dQ = (dmuScores @ K) * scale
            dK = (dmuScores.T @ Q) * scale
            dV = z.T @ dmuOutput

            gc = self.grad_clip
            self.optWo.step(clip_norm(dWo, gc))
            self.optBo.step(clip_norm(dbo, gc))
            self.optWq.step(clip_norm(X.T @ dQ, gc))
            self.optWk.step(clip_norm(X.T @ dK, gc))
            self.optWv.step(clip_norm(X.T @ dV, gc))

        return loss_bits, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="tiny_shakespeare.txt")
    ap.add_argument("--deepity-build", default=_DEEPITY_BUILD_DEFAULT)
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--embed-dim", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--settle-steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--inference-rate", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    args = ap.parse_args()

    sys.path.insert(0, args.deepity_build)
    import deepity as pd

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    stream, vocab_size = char_stream(text)
    stream = stream[: args.tokens] if args.tokens > 0 else stream

    model = AttentionProto(pd, vocab_size, embed_dim=args.embed_dim, head_dim=args.head_dim,
                           window=args.window, settle_steps=args.settle_steps,
                           inference_rate=args.inference_rate, lr=args.lr, seed=args.seed)

    hits, total_bits, n = 0, 0.0, 0
    baseline = np.log2(vocab_size)
    for i in range(args.window, len(stream)):
        window_tokens = stream[i - args.window: i]
        target = int(stream[i])
        loss_bits, pred = model.step(window_tokens, target)
        hits += int(pred == target)
        total_bits += loss_bits
        n += 1
        if n % args.log_every == 0:
            print(f"token {n} | hit rate {hits / n:.2%} | bits/char {total_bits / n:.3f} "
                  f"| uniform baseline {baseline:.3f}")

    print(f"\n[Result] {n} predictions | hit rate {hits / n:.2%} | "
          f"bits/char {total_bits / n:.3f} | uniform baseline {baseline:.3f}")


if __name__ == "__main__":
    main()
