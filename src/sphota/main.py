"""Sphota: continuous-learning language model -- PCN + associative fast-cache.

Two predictors run side by side on one stream, and their probabilities are mixed:

  * FastCache: nearest-neighbour memory keyed on the raw context
    (previous-token embedding + leaky traces). It votes over the next tokens
    of its top-k most similar past contexts -> a distribution over the vocab.
  * Network (pydeepity FullPCN with a cross-entropy head, or --model-type mlp: a
    plain PyTorch MLP of the same shape trained online with Adam): logits over the
    vocab from the same context, trained online one token at a time. Its token
    embeddings are learned online too (unless --freeze-embeddings). The cache keeps
    keying on a frozen copy of the initial embeddings, because keys built from drifting
    embeddings go stale. Learned-embedding gradients are exact (autograd) for the
    MLP and a DFA-style fixed random projection of the output error for the PCN.

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
import time
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
N_GATE_FEATURES = 9


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

    def vote(self, query: torch.Tensor, k: int, temp: float = 0.0
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (vote distribution, predicted token, top-k cosine scores), all on GPU.

        The distribution is the weighted share of the top-k neighbours voting for each
        token; weights are exp((score - best) / temp), or uniform when temp <= 0. It is
        unsmoothed, so tokens nobody voted for get exactly zero.
        The prediction is the most-voted token.
        With an empty cache the distribution is uniform, the prediction is token 0 and
        the scores are zero.
        """
        if self.size == 0:
            uniform = torch.full((self.vocab_size,), 1.0 / self.vocab_size, device="cuda")
            return uniform, torch.zeros((), dtype=torch.long, device="cuda"), torch.zeros(k, device="cuda")

        q = F.normalize(query, p=2, dim=1).squeeze(0)
        scores = self.keys[: self.size] @ q

        want_k, k = k, min(k, self.size)
        top_scores, top = torch.topk(scores, k)                 # most similar first
        candidates = self.next_tokens[top]
        weights = (torch.exp((top_scores - top_scores[0]) / temp) if temp > 0
                   else torch.ones_like(top_scores))
        votes = torch.zeros(self.vocab_size, device="cuda").index_add_(0, candidates, weights)
        votes = votes / votes.sum()
        return votes, torch.argmax(votes), F.pad(top_scores, (0, want_k - k))  # always want_k scores

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

    def __init__(self, char_len: np.ndarray, net_label: str = "pcn"):
        self.char_len = char_len
        self.net_label = net_label
        self.tps: list[tuple[int, float]] = []      # (step, tokens/sec over the block ending there)
        self.n = 0
        self.chars = 0
        self.hits = {name: 0 for name in self.names()}
        self.hit_chars = {name: 0 for name in self.names()}
        self.bits = {name: 0.0 for name in self.names() if name != "cache"}
        self.lam_sum = {name: 0.0 for name in ROUTERS}
        self.agree = {"both": 0, "cache_only": 0, "pcn_only": 0, "neither": 0}
        self.lam_sum_when = {(name, k): 0.0 for name in ROUTERS for k in self.agree}

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
        key = "both" if c and p else "cache_only" if c else "pcn_only" if p else "neither"
        self.agree[key] += 1
        for name, lam in lams.items():
            self.lam_sum[name] += lam
            self.lam_sum_when[(name, key)] += lam

    def rate(self, name: str) -> float:
        return self.hits[name] / max(self.n, 1)

    def bpc(self, name: str) -> float:
        return self.bits[name] / max(self.chars, 1)

    def log_line(self, step: int, live_lam: float, tps: float) -> str:
        return (f"Token {step} | {tps:.0f} tok/s | cache {self.rate('cache'):.2%} | "
                f"{self.net_label} {self.rate('pcn'):.2%} | "
                f"mix(lam={live_lam}) {self.rate(f'mix{live_lam}'):.2%} | "
                f"adaptive {self.rate('adaptive'):.2%} | "
                f"bpc mix {self.bpc(f'mix{live_lam}'):.3f} adaptive {self.bpc('adaptive'):.3f}")

    def result(self) -> dict[str, float]:
        """Headline numbers for sweeps: hit rates, bits/char, and the oracle ceiling."""
        fixed = [f"mix{lam}" for lam in LAM_GRID if 0 < lam < 1]
        best_fixed = min(fixed, key=self.bpc)
        return {
            "cache_hit": self.rate("cache"),
            "pcn_hit": self.rate("pcn"),
            "best_fixed": best_fixed,
            "best_fixed_hit": self.rate(best_fixed),
            "best_fixed_bpc": self.bpc(best_fixed),
            "route_bpc": self.bpc("route"),
            "adaptive_hit": self.rate("adaptive"),
            "adaptive_bpc": self.bpc("adaptive"),
            "ceiling": 1.0 - self.agree["neither"] / max(self.n, 1),
        }

    def summary(self) -> str:
        lines = [f"\n[Result] {self.n} predictions over {self.chars} characters",
                 f"{'predictor':<18}{'token hit':>11}{'chars covered':>16}{'bits/char':>12}"]
        rows = [("cache only", "cache", None), (f"{self.net_label} only", "pcn", "mix1.0")]
        rows += [(f"mix lam={lam}", f"mix{lam}", f"mix{lam}" if lam > 0 else None) for lam in LAM_GRID]
        rows += [("route (no learn)", "route", "route"), ("adaptive gate", "adaptive", "adaptive")]
        for label, name, bits_key in rows:
            bpc = f"{self.bpc(bits_key):.3f}" if bits_key is not None else "--"
            lines.append(f"{label:<18}{self.rate(name):>11.2%}"
                         f"{self.hit_chars[name] / max(self.chars, 1):>16.2%}{bpc:>12}")
        total = max(self.n, 1)
        for name in ROUTERS:
            by_case = " | ".join(f"{k} {self.lam_sum_when[(name, k)] / max(v, 1):.3f}" for k, v in self.agree.items())
            lines.append(f"{name}: mean lam {self.lam_sum[name] / max(self.n, 1):.3f} | {by_case}")
        lines.append(f"cache vs {self.net_label}: " + ", ".join(f"{k} {v / total:.2%}" for k, v in self.agree.items()))
        lines.append(f"ceiling for any cache/{self.net_label} combination (either right): "
                     f"{(total - self.agree['neither']) / total:.2%}")
        if self.tps:
            (_, first), (_, last) = self.tps[0], self.tps[-1]
            lines.append(f"throughput: first block {first:.0f} tok/s | last block {last:.0f} tok/s | "
                         f"last/first {last / first:.2f} (first block includes warm-up)")
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
        cache_temp: float = 0.05,
        cache_eps: float = 0.02,
        seed: int = 0,
        model_type: str = "pcn",
        learn_embeddings: bool = False,
        embed_lr: float = 3.3e-5,
        pcn_epc: bool = False,
        pcn_inference_rate: float = 0.1,
    ):
        torch.manual_seed(seed)
        if model_type not in ("pcn", "mlp"):
            raise ValueError(f"model_type must be 'pcn' or 'mlp', got {model_type!r}")
        self.model_type = model_type
        self.learn_embeddings = learn_embeddings
        self.pcn_epc = pcn_epc
        self.pcn_inference_rate = pcn_inference_rate
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.trace_decays = trace_decays
        self.top_k = top_k
        self.pcn_sees_cache = pcn_sees_cache
        self.gate_lr = gate_lr
        self.cache_temp = cache_temp
        self.cache_eps = cache_eps

        # Gate weights over the features in gate_features (last one is the bias).
        # Zero init -> lam = 0.5 until the gate has seen evidence.
        self.gate_w = torch.zeros(N_GATE_FEATURES, device="cuda")

        # Token codes the network reads. They are learned online when learn_embeddings is
        # set; the cache always keys on a frozen copy of the initial codes so stored keys
        # never go stale.
        self.embedding = nn.Embedding(vocab_size, embed_dim).cuda()
        self.cache_table = self.embedding.weight.detach().clone()
        self.embedding.weight.requires_grad_(learn_embeddings)
        if learn_embeddings:
            self.embed_opt = torch.optim.Adam([self.embedding.weight], lr=embed_lr)
        if learn_embeddings and model_type == "pcn":
            # The PCN returns no input gradient, so its embeddings get a DFA-style update:
            # a fixed random projection of the output error onto the embedding space.
            gen = torch.Generator(device="cuda").manual_seed(seed)
            self.dfa_proj = torch.randn(vocab_size, embed_dim, device="cuda", generator=gen) / np.sqrt(vocab_size)

        ctx_dim = embed_dim * (1 + len(trace_decays))
        net_in_dim = ctx_dim + (embed_dim if pcn_sees_cache else 0)

        if model_type == "pcn":
            self.pcn = pd.FullPCN(
                Linear(net_in_dim, latent_dim),
                ReLU(),
                Linear(latent_dim, vocab_size),
                batch_size=1,
                device="cuda"
            )
            self.pcn.configure(
                learning_rate=learning_rate,
                inference_rate=pcn_inference_rate,
                feedback_rate=1e-4,
                lmbda=0.0,
                optimizer="ADAMW",
                use_epc=pcn_epc,
                use_cross_entropy=True,
                seed=seed,
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(net_in_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, vocab_size),
            ).cuda()
            self.net_opt = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)

        self.cache = FastCache(ctx_dim, vocab_size, max_size=cache_size)

    def gate_features(self, votes: torch.Tensor, p_pcn: torch.Tensor, top_scores: torch.Tensor,
                      cache_pred: torch.Tensor) -> torch.Tensor:
        """Confidence and cross-check features for the gate, roughly in [0, 1].

        votes is the raw (unsmoothed) cache vote distribution.
        """
        def entropy(p: torch.Tensor, norm: float) -> torch.Tensor:
            return -(p * torch.log(p.clamp_min(1e-12))).sum() / norm

        return torch.stack([
            votes.max(),                                            # vote consensus
            entropy(votes, np.log(max(self.top_k, 2))),             # votes live on <= top_k tokens
            top_scores[0],                                          # best cosine match
            10.0 * (top_scores[0] - top_scores[-1]),                # drop-off from 1st to k-th match
            p_pcn.max(),
            entropy(p_pcn, np.log(self.vocab_size)),
            p_pcn[cache_pred],                                      # does the PCN back the cache's guess?
            (torch.argmax(p_pcn) == cache_pred).float(),            # hard argmax agreement
            torch.ones((), device="cuda"),
        ])

    def stream_learn(self, stream: np.ndarray, decode_map: dict[int, str],
                     live_lam: float = 0.5, log_every: int = 10000,
                     verbose: bool = True) -> StreamMetrics:
        if verbose:
            print(f"[Info] Streaming {len(stream)} tokens | trace_decays={self.trace_decays} | "
                  f"top_k={self.top_k} | pcn_sees_cache={self.pcn_sees_cache} | cache_size={self.cache.max_size} | "
                  f"gate_lr={self.gate_lr} | cache_temp={self.cache_temp} | cache_eps={self.cache_eps}\n"
                  f"       model={self.model_type} | learn_embeddings={self.learn_embeddings} | "
                  f"pcn_epc={self.pcn_epc} | pcn_inference_rate={self.pcn_inference_rate}")

        char_len = np.array([len(decode_map[t]) for t in range(self.vocab_size)], dtype=np.int64)
        metrics = StreamMetrics(char_len, net_label=self.model_type)
        # Leaky traces of the frozen codes (cache keys) and, when embeddings are learned,
        # of the learned codes as they were when each token went by (network input).
        cache_traces = [torch.zeros(1, self.embed_dim, device="cuda") for _ in self.trace_decays]
        net_traces = [torch.zeros(1, self.embed_dim, device="cuda") for _ in self.trace_decays]
        one_hot = np.zeros((1, self.vocab_size), dtype=np.float32)
        block_start, block_step = time.perf_counter(), 0

        for i in range(1, len(stream)):
            prev_token = int(stream[i - 1])
            current_token = int(stream[i])

            cache_prev = self.cache_table[prev_token].unsqueeze(0)
            context = torch.cat([cache_prev, *cache_traces], dim=1)

            # Cache vote
            votes, cache_pred, top_scores = self.cache.vote(context, self.top_k, self.cache_temp)
            p_cache = (1.0 - self.cache_eps) * votes + self.cache_eps / self.vocab_size

            # Network logits. The learned embedding row is the only part of the input that
            # carries a gradient; traces are constants.
            if self.learn_embeddings:
                net_prev = self.embedding.weight[prev_token].unsqueeze(0)
                net_prev_val = net_prev.detach().clone()
                net_input = torch.cat([net_prev, *net_traces], dim=1)
            else:
                net_prev_val = cache_prev
                net_input = context
            if self.pcn_sees_cache:
                net_input = torch.cat([net_input, self.cache_table[cache_pred].unsqueeze(0)], dim=1)

            if self.model_type == "mlp":
                logits = self.mlp(net_input)
                p_pcn = F.softmax(logits.detach(), dim=1).squeeze(0)
            else:
                # 0 settling steps = plain feed-forward readout
                x_np = net_input.detach().cpu().numpy().astype(np.float32)
                logits_np = self.pcn.predict(x_np, inference_steps=0).reshape(1, -1).astype(np.float32)
                p_pcn = F.softmax(torch.from_numpy(logits_np).cuda(), dim=1).squeeze(0)

            # Routers: per-token lam from cache/PCN confidence
            feats = self.gate_features(votes, p_pcn, top_scores, cache_pred)
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
            if self.model_type == "mlp":
                target = torch.tensor([current_token], device="cuda")
                self.net_opt.zero_grad()
                if self.learn_embeddings:
                    self.embed_opt.zero_grad()
                F.cross_entropy(logits, target).backward()
                self.net_opt.step()
                if self.learn_embeddings:
                    self.embed_opt.step()
            else:
                one_hot[:] = 0.0
                one_hot[0, current_token] = 1.0
                self.pcn.train_step(x_np, one_hot, inference_steps=1)
                if self.learn_embeddings:
                    err = p_pcn.clone()
                    err[current_token] -= 1.0                       # softmax - onehot
                    grad = torch.zeros_like(self.embedding.weight)
                    grad[prev_token] = err @ self.dfa_proj
                    self.embedding.weight.grad = grad
                    self.embed_opt.step()

            # Roll the leaky integrators (they summarise tokens before prev_token)
            cache_traces = [d * t + (1.0 - d) * cache_prev for d, t in zip(self.trace_decays, cache_traces)]
            if self.learn_embeddings:
                net_traces = [d * t + (1.0 - d) * net_prev_val for d, t in zip(self.trace_decays, net_traces)]

            if log_every and i % log_every == 0:
                now = time.perf_counter()
                tps = (i - block_step) / (now - block_start)
                metrics.tps.append((i, tps))
                block_start, block_step = now, i
                if verbose:
                    guess = decode_map.get(preds[f"mix{live_lam}"], "?")
                    actual = decode_map.get(current_token, "?")
                    print(f"{metrics.log_line(i, live_lam, tps)} | guess {guess!r} | actual {actual!r}")

        if verbose:
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
    ap.add_argument("--start", type=int, default=0, help="first token of the stream segment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--trace-decays", type=str, default="0.5,0.9")
    ap.add_argument("--model-type", choices=["pcn", "mlp"], default="pcn",
                    help="pcn = pydeepity FullPCN; mlp = same-shape PyTorch MLP trained online with Adam")
    ap.add_argument("--pcn-epc", action="store_true",
                    help="error-based PC: each settling step is a full forward+backward sweep (diagnostic; "
                         "not layer-local, so it is not the DKP O(1)-depth setting)")
    ap.add_argument("--pcn-inference-rate", type=float, default=0.1, help="PCN settling rate")
    ap.add_argument("--lr", type=float, default=3.3e-5, help="network learning rate")
    ap.add_argument("--embed-lr", type=float, default=3.3e-5, help="learning rate of the learned token embeddings")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="keep the token codes fixed random (reproduces the earlier frozen runs; use --lr 1e-3)")
    ap.add_argument("--cache-temp", type=float, default=0.05,
                    help="similarity weighting of cache votes, exp(score/temp); 0 = plain counts")
    ap.add_argument("--cache-eps", type=float, default=0.02, help="uniform smoothing mass on the cache distribution")
    ap.add_argument("--gate-lr", type=float, default=0.05, help="SGD rate of the adaptive lam gate")
    ap.add_argument("--lam", type=float, default=0.5, choices=LAM_GRID, help="mixture weight shown live")
    ap.add_argument("--pcn-sees-cache", action="store_true",
                    help="also feed the cache's guess into the PCN (default off, keeps the two independent)")
    ap.add_argument("--log-every", type=int, default=10000)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    text = load_text()
    if args.tokenizer == "bpe":
        stream, vocab_size, decode_map = bpe_stream(text, args.bpe_vocab)
    else:
        stream, vocab_size, decode_map = char_stream(text)

    stream = stream[args.start: args.start + args.tokens if args.tokens > 0 else None]

    model = SphotaLM(
        vocab_size=vocab_size,
        trace_decays=tuple(float(d) for d in args.trace_decays.split(",")),
        top_k=args.top_k,
        learning_rate=args.lr,
        pcn_sees_cache=args.pcn_sees_cache,
        cache_size=len(stream),
        gate_lr=args.gate_lr,
        cache_temp=args.cache_temp,
        cache_eps=args.cache_eps,
        seed=args.seed,
        model_type=args.model_type,
        learn_embeddings=not args.freeze_embeddings,
        embed_lr=args.embed_lr,
        pcn_epc=args.pcn_epc,
        pcn_inference_rate=args.pcn_inference_rate,
    )
    model.stream_learn(stream, decode_map, live_lam=args.lam, log_every=args.log_every)