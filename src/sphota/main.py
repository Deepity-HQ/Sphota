"""Sphota: continuous-learning language model -- PCN + associative fast-cache.

Two predictors run side by side on one stream, and their probabilities are mixed:

  * FastCache: nearest-neighbour memory keyed on the raw context
    (previous-token embedding + leaky traces). It votes over the next tokens
    of its top-k most similar past contexts -> a distribution over the vocab.
  * PCN (pydeepity FullPCN, cross-entropy head): logits over the vocab from the
    same context, trained online one token at a time.

  p_mix = lam * p_pcn + (1 - lam) * p_cache

lam is swept over a fixed grid, and also set per token by two routers:
  * route:    lam = 1 - (largest share of the top-k votes), no learning.
  * adaptive: lam = sigmoid(w . f), where f holds the cache's vote consensus and
              entropy and the PCN's own confidence. w is learned online by SGD on
              the mixture's log-loss, so the gate directly minimises bits/char.

Metrics are reported for the cache alone, the PCN alone, and the mixture at
several lam values, as token hit rate, share of characters inside correctly
predicted tokens, and bits per character (tokenizer-independent).

Usage:
    python -m sphota.main                                  # chars, full stream
    python -m sphota.main --tokenizer bpe --bpe-vocab 512  # small BPE
    python -m sphota.main --tokens 50000                   # first 50k tokens only
"""

import argparse
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pydeepity as pd
from pydeepity.layer import Linear, ReLU

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "tiny_shakespeare.txt"
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ROUTERS = ("route", "adaptive")
N_GATE_FEATURES = 5


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_text(path: str = DATA_PATH) -> str:
    if not os.path.exists(path):
        print("Downloading raw Tiny Shakespeare text...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def char_stream(text: str) -> tuple[np.ndarray, int, dict[int, str]]:
    chars = sorted(set(text))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    decode_map = {i: ch for i, ch in enumerate(chars)}
    stream = np.array([char_to_int[c] for c in text], dtype=np.int32)
    print(f"Loaded Tiny Shakespeare (chars): {len(stream)} tokens, {len(chars)} vocab size")
    return stream, len(chars), decode_map


def bpe_stream(text: str, target_vocab: int) -> tuple[np.ndarray, int, dict[int, str]]:
    """Byte-level BPE trained on the corpus itself (cached to disk after the first run)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok_path = f"shakespeare_bpe_{target_vocab}.json"
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=target_vocab,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator([text], trainer)
        tokenizer.save(tok_path)

    stream = np.array(tokenizer.encode(text).ids, dtype=np.int32)
    vocab_size = tokenizer.get_vocab_size()
    decode_map = {i: tokenizer.decode([i]) for i in range(vocab_size)}
    print(f"Loaded Tiny Shakespeare (BPE): {len(stream)} tokens, {vocab_size} vocab size, "
          f"{len(text) / len(stream):.2f} chars/token")
    return stream, vocab_size, decode_map


# ----------------------------------------------------------------------------
# Associative memory
# ----------------------------------------------------------------------------

class FastCache:
    """Pre-allocated GPU memory of (normalized context key -> next token), with top-k voting."""

    def __init__(self, key_dim: int, vocab_size: int, max_size: int):
        self.vocab_size = vocab_size
        self.max_size = max_size
        self.keys = torch.zeros(max_size, key_dim, device="cuda")  # stored already L2-normalized
        self.next_tokens = torch.zeros(max_size, dtype=torch.long, device="cuda")
        self.size = 0
        self.ptr = 0

    def vote(self, query: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (distribution over vocab, predicted token), both on GPU.

        The distribution is the share of the top-k neighbours voting for each token.
        The prediction is the most-voted token, ties broken toward the most similar key.
        With an empty cache the distribution is uniform and the prediction is token 0.
        """
        if self.size == 0:
            uniform = torch.full((self.vocab_size,), 1.0 / self.vocab_size, device="cuda")
            return uniform, torch.zeros((), dtype=torch.long, device="cuda")

        q = F.normalize(query, p=2, dim=1).squeeze(0)
        scores = self.keys[: self.size] @ q

        k = min(k, self.size)
        top = torch.topk(scores, k).indices                     # most similar first
        candidates = self.next_tokens[top]
        counts = torch.bincount(candidates, minlength=self.vocab_size).float()
        pred = candidates[torch.argmax(counts[candidates])]     # first max = most similar
        return counts / k, pred

    def imprint(self, key: torch.Tensor, next_token: int) -> None:
        self.keys[self.ptr] = F.normalize(key, p=2, dim=1).squeeze(0)
        self.next_tokens[self.ptr] = next_token
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

class StreamMetrics:
    """Running accuracies, character coverage, agreement breakdown and bits/char."""

    def __init__(self, char_len: np.ndarray):
        self.char_len = char_len
        self.n = 0
        self.chars = 0
        self.hits = {name: 0 for name in self.names()}
        self.hit_chars = {name: 0 for name in self.names()}
        self.bits = {name: 0.0 for name in self.names() if name != "cache"}
        self.lam_sum = {name: 0.0 for name in ROUTERS}
        self.lam_sum_when = {(name, ok): 0.0 for name in ROUTERS for ok in (True, False)}
        self.n_when = {True: 0, False: 0}
        self.agree = {"both": 0, "cache_only": 0, "pcn_only": 0, "neither": 0}

    @staticmethod
    def names() -> list[str]:
        return ["cache", "pcn"] + [f"mix{lam}" for lam in LAM_GRID] + list(ROUTERS)

    def update(self, actual: int, preds: dict[str, int], p_true: dict[str, float],
               lams: dict[str, float]) -> None:
        n_chars = int(self.char_len[actual])
        self.n += 1
        self.chars += n_chars

        for name, pred in preds.items():
            if pred == actual:
                self.hits[name] += 1
                self.hit_chars[name] += n_chars

        for name, p in p_true.items():
            self.bits[name] -= np.log2(max(p, 1e-12))

        c, p = preds["cache"] == actual, preds["pcn"] == actual
        self.n_when[c] += 1
        for name, lam in lams.items():
            self.lam_sum[name] += lam
            self.lam_sum_when[(name, c)] += lam
        key = "both" if c and p else "cache_only" if c else "pcn_only" if p else "neither"
        self.agree[key] += 1

    def rate(self, name: str) -> float:
        return self.hits[name] / max(self.n, 1)

    def bpc(self, name: str) -> float:
        return self.bits[name] / max(self.chars, 1)

    def log_line(self, step: int, live_lam: float) -> str:
        return (f"Token {step} | cache {self.rate('cache'):.2%} | pcn {self.rate('pcn'):.2%} | "
                f"mix(lam={live_lam}) {self.rate(f'mix{live_lam}'):.2%} | "
                f"adaptive {self.rate('adaptive'):.2%} | "
                f"bpc mix {self.bpc(f'mix{live_lam}'):.3f} adaptive {self.bpc('adaptive'):.3f}")

    def summary(self) -> str:
        lines = [f"\n[Result] {self.n} predictions over {self.chars} characters",
                 f"{'predictor':<18}{'token hit':>11}{'chars covered':>16}{'bits/char':>12}"]
        rows = [("cache only", "cache", None), ("pcn only", "pcn", "mix1.0")]
        rows += [(f"mix lam={lam}", f"mix{lam}", f"mix{lam}" if lam > 0 else None) for lam in LAM_GRID]
        rows += [("route (no learn)", "route", "route"), ("adaptive gate", "adaptive", "adaptive")]
        for label, name, bits_key in rows:
            bpc = f"{self.bpc(bits_key):.3f}" if bits_key is not None else "--"
            lines.append(f"{label:<18}{self.rate(name):>11.2%}"
                         f"{self.hit_chars[name] / max(self.chars, 1):>16.2%}{bpc:>12}")
        total = max(self.n, 1)
        for name in ROUTERS:
            lines.append(f"{name}: mean lam {self.lam_sum[name] / max(self.n, 1):.3f} | "
                         f"when cache right {self.lam_sum_when[(name, True)] / max(self.n_when[True], 1):.3f} | "
                         f"when cache wrong {self.lam_sum_when[(name, False)] / max(self.n_when[False], 1):.3f}")
        lines.append("cache vs pcn: " + ", ".join(f"{k} {v / total:.2%}" for k, v in self.agree.items()))
        lines.append(f"ceiling for any cache/pcn combination (either right): "
                     f"{(total - self.agree['neither']) / total:.2%}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class SphotaLM:
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        latent_dim: int = 256,
        trace_decays: tuple[float, ...] = (0.5, 0.9),
        top_k: int = 5,
        learning_rate: float = 1e-3,
        pcn_sees_cache: bool = False,
        cache_size: int = 50000,
        gate_lr: float = 0.05,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.trace_decays = trace_decays
        self.top_k = top_k
        self.pcn_sees_cache = pcn_sees_cache
        self.gate_lr = gate_lr

        # Gate weights over [cache max share, cache entropy, pcn max prob, pcn entropy, bias].
        # Zero init -> lam = 0.5 until the gate has seen evidence.
        self.gate_w = torch.zeros(N_GATE_FEATURES, device="cuda")

        # Fixed random token codes (never trained).
        self.embedding = nn.Embedding(vocab_size, embed_dim).cuda()
        self.emb_table = self.embedding.weight.detach()

        ctx_dim = embed_dim * (1 + len(trace_decays))
        pcn_in_dim = ctx_dim + (embed_dim if pcn_sees_cache else 0)

        self.pcn = pd.FullPCN(
            Linear(pcn_in_dim, latent_dim),
            ReLU(),
            Linear(latent_dim, vocab_size),
            batch_size=1,
            device="cuda"
        )
        self.pcn.configure(
            learning_rate=learning_rate,
            inference_rate=0.1,
            feedback_rate=1e-4,
            lmbda=0.0,
            optimizer="ADAMW",
            use_cross_entropy=True,
            seed=0,
        )

        self.cache = FastCache(ctx_dim, vocab_size, max_size=cache_size)

    def gate_features(self, p_cache: torch.Tensor, p_pcn: torch.Tensor) -> torch.Tensor:
        """Confidence features for the gate, all in [0, 1]."""
        def entropy(p: torch.Tensor, norm: float) -> torch.Tensor:
            return -(p * torch.log(p.clamp_min(1e-12))).sum() / norm

        return torch.stack([
            p_cache.max(),
            entropy(p_cache, np.log(max(self.top_k, 2))),   # votes live on <= top_k tokens
            p_pcn.max(),
            entropy(p_pcn, np.log(self.vocab_size)),
            torch.ones((), device="cuda"),
        ])

    def stream_learn(self, stream: np.ndarray, decode_map: dict[int, str],
                     live_lam: float = 0.5, log_every: int = 10000) -> StreamMetrics:
        print(f"[Info] Streaming {len(stream)} tokens | trace_decays={self.trace_decays} | "
              f"top_k={self.top_k} | pcn_sees_cache={self.pcn_sees_cache} | cache_size={self.cache.max_size} | "
              f"gate_lr={self.gate_lr}")

        char_len = np.array([len(decode_map[t]) for t in range(self.vocab_size)], dtype=np.int64)
        metrics = StreamMetrics(char_len)
        traces = [torch.zeros(1, self.embed_dim, device="cuda") for _ in self.trace_decays]
        one_hot = np.zeros((1, self.vocab_size), dtype=np.float32)

        for i in range(1, len(stream)):
            prev_token = int(stream[i - 1])
            current_token = int(stream[i])

            prev_emb = self.emb_table[prev_token].unsqueeze(0)
            context = torch.cat([prev_emb, *traces], dim=1)

            # Cache vote
            p_cache, cache_pred = self.cache.vote(context, self.top_k)

            # PCN logits (0 settling steps = plain feed-forward readout)
            pcn_input = context
            if self.pcn_sees_cache:
                pcn_input = torch.cat([context, self.emb_table[cache_pred].unsqueeze(0)], dim=1)
            x_np = pcn_input.cpu().numpy().astype(np.float32)

            logits_np = self.pcn.predict(x_np, inference_steps=0).reshape(1, -1).astype(np.float32)
            p_pcn = F.softmax(torch.from_numpy(logits_np).cuda(), dim=1).squeeze(0)

            # Routers: per-token lam from cache/PCN confidence
            feats = self.gate_features(p_cache, p_pcn)
            lam_route = 1.0 - feats[0]
            lam_gate = torch.sigmoid(self.gate_w @ feats)

            # Mixtures, gathered in one GPU->CPU transfer
            lams = [*LAM_GRID, lam_route, lam_gate]
            mixes = [lam * p_pcn + (1.0 - lam) * p_cache for lam in lams]
            floor = 1e-4 / self.vocab_size
            probs = [(1 - 1e-4) * m[current_token] + floor for m in mixes]
            stacked = torch.stack(
                [cache_pred.float(), torch.argmax(p_pcn).float()]
                + [torch.argmax(m).float() for m in mixes]
                + [p for j, p in enumerate(probs) if not (j < len(LAM_GRID) and LAM_GRID[j] == 0)]
                + [lam_route, lam_gate]
            ).tolist()

            names = [f"mix{lam}" for lam in LAM_GRID] + list(ROUTERS)
            n_mix = len(names)
            preds = {"cache": int(stacked[0]), "pcn": int(stacked[1])}
            preds.update({name: int(stacked[2 + j]) for j, name in enumerate(names)})
            scored = [name for j, name in enumerate(names) if not (j < len(LAM_GRID) and LAM_GRID[j] == 0)]
            p_true = {name: stacked[2 + n_mix + j] for j, name in enumerate(scored)}
            lam_vals = dict(zip(ROUTERS, stacked[-2:]))
            metrics.update(current_token, preds, p_true, lam_vals)

            # Gate SGD step on -log p_mix(true): dL/dz = -(pp - pc) / p_mix * lam * (1 - lam)
            p_c, p_p = p_cache[current_token], p_pcn[current_token]
            p_m = (lam_gate * p_p + (1.0 - lam_gate) * p_c).clamp_min(1e-12)
            grad_z = -(p_p - p_c) / p_m * lam_gate * (1.0 - lam_gate)
            self.gate_w -= self.gate_lr * grad_z * feats

            # Learn
            self.cache.imprint(context, current_token)
            one_hot[:] = 0.0
            one_hot[0, current_token] = 1.0
            self.pcn.train_step(x_np, one_hot, inference_steps=1)

            # Roll the leaky integrators (they summarise tokens before prev_token)
            traces = [d * t + (1.0 - d) * prev_emb for d, t in zip(self.trace_decays, traces)]

            if i % log_every == 0:
                guess = decode_map.get(preds[f"mix{live_lam}"], "?")
                actual = decode_map.get(current_token, "?")
                print(f"{metrics.log_line(i, live_lam)} | guess {guess!r} | actual {actual!r}")

        print(metrics.summary())
        return metrics


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sphota continuous-learning stream")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--bpe-vocab", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=0, help="stream length (0 = whole corpus)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--trace-decays", type=str, default="0.5,0.9")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate-lr", type=float, default=0.05, help="SGD rate of the adaptive lam gate")
    ap.add_argument("--lam", type=float, default=0.5, choices=LAM_GRID, help="mixture weight shown live")
    ap.add_argument("--pcn-sees-cache", action="store_true",
                    help="also feed the cache's guess into the PCN (default off, keeps the two independent)")
    ap.add_argument("--log-every", type=int, default=10000)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)

    text = load_text()
    if args.tokenizer == "bpe":
        stream, vocab_size, decode_map = bpe_stream(text, args.bpe_vocab)
    else:
        stream, vocab_size, decode_map = char_stream(text)

    if args.tokens > 0:
        stream = stream[: args.tokens]

    model = SphotaLM(
        vocab_size=vocab_size,
        trace_decays=tuple(float(d) for d in args.trace_decays.split(",")),
        top_k=args.top_k,
        learning_rate=args.lr,
        pcn_sees_cache=args.pcn_sees_cache,
        cache_size=len(stream),
        gate_lr=args.gate_lr,
    )
    model.stream_learn(stream, decode_map, live_lam=args.lam, log_every=args.log_every)