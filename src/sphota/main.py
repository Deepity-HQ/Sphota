"""Sphota: continuous-learning language model -- PCN + associative fast-cache.

Two predictors run side by side on one stream, and their probabilities are mixed:

  * FastCache: nearest-neighbour memory keyed on the raw context
    (previous-token embedding + leaky traces). It votes over the next tokens
    of its top-k most similar past contexts -> a distribution over the vocab.
  * Network (deepity FullPCN with a cross-entropy head, or --model-type mlp: a
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

from sphota.cache_bench import HashedInterpNgram, InterpNgram

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "tiny_shakespeare.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # the PCN needs cuda; the mlp runs anywhere
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ROUTERS = ("route", "adaptive")
N_GATE_FEATURES = 9


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_PATH = "enwik8_10M.txt"


def fetch_enwik8(path: str = ENWIK8_PATH, n_bytes: int = 10_000_000) -> None:
    """First n_bytes of enwik8 (Wikipedia XML), decoded to text. The zip member is read in
    memory, never extracted to disk, and sanity-checked before use."""
    import io
    import zipfile

    print(f"Downloading enwik8 (36 MB) and keeping its first {n_bytes} bytes...")
    with urllib.request.urlopen(ENWIK8_URL, timeout=120) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    raw = archive.read("enwik8")
    if len(raw) != 100_000_000 or not raw.startswith(b"<mediawiki"):
        raise RuntimeError("unexpected enwik8 contents; refusing to use the download")
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw[:n_bytes].decode("utf-8", errors="ignore"))


def load_text(path: str = DATA_PATH) -> str:
    if path == "enwik8":
        path = ENWIK8_PATH
        if not os.path.exists(path):
            fetch_enwik8(path)
    elif not os.path.exists(path):
        print("Downloading raw Tiny Shakespeare text...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def char_stream(text: str) -> tuple[np.ndarray, int, dict[int, str]]:
    chars = sorted(set(text))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    decode_map = {i: ch for i, ch in enumerate(chars)}
    stream = np.array([char_to_int[c] for c in text], dtype=np.int32)
    print(f"Loaded text (chars): {len(stream)} tokens, {len(chars)} vocab size")
    return stream, len(chars), decode_map


def byte_stream(text: str) -> tuple[np.ndarray, int, dict[int, str]]:
    """UTF-8 bytes as tokens (vocab 256): the usual choice for enwik8, one token = one byte."""
    stream = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    decode_map = {i: chr(i) if 32 <= i < 127 else "." for i in range(256)}   # one char per byte keeps bits/char = bits/byte
    print(f"Loaded text (bytes): {len(stream)} tokens, 256 vocab size")
    return stream, 256, decode_map


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
        self.keys = torch.zeros(max_size, key_dim, device=DEVICE)  # stored already L2-normalized
        self.next_tokens = torch.zeros(max_size, dtype=torch.long, device=DEVICE)
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
            uniform = torch.full((self.vocab_size,), 1.0 / self.vocab_size, device=DEVICE)
            return uniform, torch.zeros((), dtype=torch.long, device=DEVICE), torch.zeros(k, device=DEVICE)

        q = F.normalize(query, p=2, dim=1).squeeze(0)
        scores = self.keys[: self.size] @ q

        want_k, k = k, min(k, self.size)
        top_scores, top = torch.topk(scores, k)                 # most similar first
        candidates = self.next_tokens[top]
        weights = (torch.exp((top_scores - top_scores[0]) / temp) if temp > 0
                   else torch.ones_like(top_scores))
        votes = torch.zeros(self.vocab_size, device=DEVICE).index_add_(0, candidates, weights)
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

    def __init__(self, char_len: np.ndarray, net_label: str = "pcn", mem_label: str = "cache"):
        self.char_len = char_len
        self.net_label = net_label
        self.mem_label = mem_label
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
        return (f"Token {step} | {tps:.0f} tok/s | {self.mem_label} {self.rate('cache'):.2%} | "
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
        rows = [(f"{self.mem_label} only", "cache", None), (f"{self.net_label} only", "pcn", "mix1.0")]
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
        lines.append(f"{self.mem_label} vs {self.net_label}: " + ", ".join(f"{k} {v / total:.2%}" for k, v in self.agree.items()))
        lines.append(f"ceiling for any {self.mem_label}/{self.net_label} combination (either right): "
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
    """Memory expert + online network, mixed by a learned gate.

    memory="ngram": exact Witten-Bell n-gram (ngram_rows_bits=0) or its fixed-size hashed
    version; the baseline floor the network has to beat. memory="cache": the earlier cosine
    cache (FastCache), kept so older runs and sweep.py still reproduce.

    The network reads the last `window` token embeddings (concatenated, newest first) plus
    leaky traces of the same embeddings.
    """

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
        pcn_feedback_rate: float = 1e-4,
        memory: str = "cache",
        ngram_order: int = 5,
        ngram_rows_bits: int = 0,
        window: int = 1,
        hidden_layers: int = 1,
        lr_halflife: int = 0,
        pcn_lmbda: float = 0.0,
        pcn_train_steps: int = 1,
        freeze_first_layer: bool = False,
        freeze_hidden: bool = False,
        pcn_class: str = "full",
        stack_memory: bool = False,
        dkp_scale: float = 10.0,
    ):
        torch.manual_seed(seed)
        if model_type not in ("pcn", "mlp"):
            raise ValueError(f"model_type must be 'pcn' or 'mlp', got {model_type!r}")
        if memory not in ("cache", "ngram"):
            raise ValueError(f"memory must be 'cache' or 'ngram', got {memory!r}")
        if pcn_class not in ("full", "dkp"):
            raise ValueError(f"pcn_class must be 'full' or 'dkp', got {pcn_class!r}")
        self.model_type = model_type
        self.memory_kind = memory
        self.learn_embeddings = learn_embeddings
        self.lr = learning_rate
        self.embed_lr = embed_lr
        self.pcn_epc = pcn_epc
        self.pcn_inference_rate = pcn_inference_rate
        self.pcn_feedback_rate = pcn_feedback_rate
        self.pcn_lmbda = pcn_lmbda
        self.pcn_train_steps = pcn_train_steps
        self.pcn_class = pcn_class
        self.stack_memory = stack_memory
        # DKPPCN has a squared-error head, so its raw outputs are scored as softmax(scale * out);
        # the scale is one scalar learned online on the log-loss.
        self.dkp_scale = dkp_scale
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.trace_decays = trace_decays
        self.top_k = top_k
        self.pcn_sees_cache = pcn_sees_cache
        self.gate_lr = gate_lr
        self.cache_temp = cache_temp
        self.cache_eps = cache_eps
        self.ngram_order = ngram_order
        self.ngram_rows_bits = ngram_rows_bits
        self.window = window
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.lr_halflife = lr_halflife
        self.mem = None                       # n-gram memory, built in stream_learn (it needs the stream)

        # Gate weights over the features in gate_features (last one is the bias).
        # Zero init -> lam = 0.5 until the gate has seen evidence.
        self.gate_w = torch.zeros(N_GATE_FEATURES, device=DEVICE)

        # Token codes the network reads. They are learned online when learn_embeddings is
        # set; the cosine cache always keys on a frozen copy of the initial codes so stored
        # keys never go stale.
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(DEVICE)
        self.cache_table = self.embedding.weight.detach().clone()
        self.embedding.weight.requires_grad_(learn_embeddings)
        if learn_embeddings:
            self.embed_opt = torch.optim.Adam([self.embedding.weight], lr=embed_lr)
        if learn_embeddings and model_type == "pcn":
            # The PCN returns no input gradient, so its embeddings get a DFA-style update:
            # a fixed random projection of the output error onto the whole input window.
            gen = torch.Generator(device=DEVICE).manual_seed(seed)
            self.dfa_proj = torch.randn(vocab_size, window * embed_dim, device=DEVICE, generator=gen) / np.sqrt(vocab_size)

        net_in_dim = embed_dim * (window + len(trace_decays)) + (embed_dim if pcn_sees_cache else 0)
        if stack_memory:
            net_in_dim += vocab_size + 2          # the memory's distribution plus its two confidence numbers

        if model_type == "pcn":
            import deepity as pd                    # the pip package `pydeepity` installs the module `deepity`
            from deepity.layer import Linear, ReLU

            layers = [Linear(net_in_dim, latent_dim), ReLU()]
            for _ in range(hidden_layers - 1):
                layers += [Linear(latent_dim, latent_dim), ReLU()]
            layers.append(Linear(latent_dim, vocab_size))
            if pcn_class == "dkp":
                # Direct Kolen-Pollack: learned feedback weights send the output error straight to
                # every hidden layer, so one settling step already trains all of them.
                self.pcn = pd.DKPPCN(*layers, batch_size=1, device=DEVICE)
                self.pcn.configure(
                    learning_rate=learning_rate,
                    inference_rate=pcn_inference_rate,
                    feedback_rate=pcn_feedback_rate,
                    lmbda=pcn_lmbda,
                    optimizer="ADAMW",
                )
            else:
                self.pcn = pd.FullPCN(*layers, batch_size=1, device=DEVICE)
                self.pcn.configure(
                    learning_rate=learning_rate,
                    inference_rate=pcn_inference_rate,
                    feedback_rate=pcn_feedback_rate,
                    lmbda=pcn_lmbda,
                    optimizer="ADAMW",
                    use_epc=pcn_epc,
                    use_cross_entropy=True,
                    seed=seed,
                )
        else:
            layers = [nn.Linear(net_in_dim, latent_dim), nn.ReLU()]
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(latent_dim, latent_dim), nn.ReLU()]
            layers.append(nn.Linear(latent_dim, vocab_size))
            self.mlp = nn.Sequential(*layers).to(DEVICE)
            if freeze_first_layer:
                # Diagnostic: emulates a PCN whose first layer never receives an error signal.
                for prm in self.mlp[0].parameters():
                    prm.requires_grad_(False)
            if freeze_hidden:
                # Diagnostic: only the readout trains (what a PCN with too few settling steps does).
                for mod in self.mlp[:-1]:
                    for prm in mod.parameters():
                        prm.requires_grad_(False)
            self.net_opt = torch.optim.Adam([q for q in self.mlp.parameters() if q.requires_grad], lr=learning_rate)

        if memory == "cache":
            ctx_dim = embed_dim * (1 + len(trace_decays))
            self.cache = FastCache(ctx_dim, vocab_size, max_size=cache_size)
            self.mem_ent_norm = float(np.log(max(top_k, 2)))      # votes live on <= top_k tokens
        else:
            self.mem_ent_norm = float(np.log(vocab_size))

    def gate_features(self, mem_dist: torch.Tensor, p_net: torch.Tensor, mem_info: torch.Tensor,
                      mem_pred: torch.Tensor) -> torch.Tensor:
        """Confidence and cross-check features for the gate, roughly in [0, 1].

        mem_dist is the memory expert's raw (unsmoothed) distribution. mem_info holds two
        memory-specific confidence numbers: for the cosine cache the best match and its
        drop-off to the k-th, for the n-gram the deepest context order used (as a fraction
        of the maximum) and how often that context has been seen (log-scaled).
        """
        def entropy(p: torch.Tensor, norm: float) -> torch.Tensor:
            return -(p * torch.log(p.clamp_min(1e-12))).sum() / norm

        return torch.stack([
            mem_dist.max(),                                         # consensus
            entropy(mem_dist, self.mem_ent_norm),
            mem_info[0],
            mem_info[1],
            p_net.max(),
            entropy(p_net, np.log(self.vocab_size)),
            p_net[mem_pred],                                        # does the network back the memory's guess?
            (torch.argmax(p_net) == mem_pred).float(),              # hard argmax agreement
            torch.ones((), device=DEVICE),
        ])

    def _memory_vote(self, i: int, context: torch.Tensor | None):
        """Returns (raw distribution, argmax, mem_info) for predicting stream[i]."""
        if self.memory_kind == "cache":
            votes, pred, top_scores = self.cache.vote(context, self.top_k, self.cache_temp)
            return votes, pred, torch.stack([top_scores[0], 10.0 * (top_scores[0] - top_scores[-1])])
        dist_np, _ = self.mem.vote(None, i)
        dist = torch.from_numpy(dist_np.astype(np.float32)).to(DEVICE)
        info = torch.tensor([self.mem.last_order / self.ngram_order,
                             min(float(np.log1p(self.mem.last_total) / np.log1p(1000.0)), 1.0)], device=DEVICE)
        return dist, torch.argmax(dist), info

    def stream_learn(self, stream: np.ndarray, decode_map: dict[int, str],
                     live_lam: float = 0.5, log_every: int = 10000,
                     verbose: bool = True) -> StreamMetrics:
        if self.memory_kind == "ngram":
            self.mem = (InterpNgram(self.ngram_order, self.vocab_size, stream) if self.ngram_rows_bits == 0
                        else HashedInterpNgram(self.ngram_order, self.vocab_size, stream, self.ngram_rows_bits))
        if verbose:
            mem_desc = (f"cache(top_k={self.top_k}, temp={self.cache_temp}, size={self.cache.max_size})"
                        if self.memory_kind == "cache" else
                        f"ngram(order={self.ngram_order}, rows_bits={self.ngram_rows_bits or 'exact'})")
            print(f"[Info] Streaming {len(stream)} tokens | memory={mem_desc} | eps={self.cache_eps} | "
                  f"gate_lr={self.gate_lr}\n"
                  f"       model={self.model_type} | window={self.window} | traces={self.trace_decays} | "
                  f"layers={self.hidden_layers}x{self.latent_dim} | learn_embeddings={self.learn_embeddings} | "
                  f"lr={self.lr} | embed_lr={self.embed_lr} | lr_halflife={self.lr_halflife or 'off'} | "
                  f"pcn_sees_cache={self.pcn_sees_cache}\n"
                  f"       pcn_epc={self.pcn_epc} | pcn_inference_rate={self.pcn_inference_rate} | "
                  f"pcn_feedback_rate={self.pcn_feedback_rate} | pcn_lmbda={self.pcn_lmbda} | pcn_train_steps={self.pcn_train_steps} | pcn_class={self.pcn_class} | stack_memory={self.stack_memory} | device={DEVICE}")

        char_len = np.array([len(decode_map[t]) for t in range(self.vocab_size)], dtype=np.int64)
        mem_label = "cache" if self.memory_kind == "cache" else "ngram"
        metrics = StreamMetrics(char_len, net_label=self.model_type, mem_label=mem_label)
        # Leaky traces over the network's embeddings (as they were when each token went by),
        # and, for the cosine cache, over the frozen codes.
        net_traces = [torch.zeros(1, self.embed_dim, device=DEVICE) for _ in self.trace_decays]
        cache_traces = [torch.zeros(1, self.embed_dim, device=DEVICE) for _ in self.trace_decays]
        one_hot = np.zeros((1, self.vocab_size), dtype=np.float32)
        block_start, block_step = time.perf_counter(), 0

        for i in range(1, len(stream)):
            prev_token = int(stream[i - 1])
            current_token = int(stream[i])

            # Memory expert
            context = None
            if self.memory_kind == "cache":
                cache_prev = self.cache_table[prev_token].unsqueeze(0)
                context = torch.cat([cache_prev, *cache_traces], dim=1)
            mem_dist, mem_pred, mem_info = self._memory_vote(i, context)
            p_cache = (1.0 - self.cache_eps) * mem_dist + self.cache_eps / self.vocab_size

            # Network input: the last `window` embeddings (newest first) plus traces. Only the
            # window rows of a learned table carry a gradient; traces are constants.
            win_idx = torch.tensor([int(stream[max(i - 1 - j, 0)]) for j in range(self.window)], device=DEVICE)
            window_emb = self.embedding.weight[win_idx] if self.learn_embeddings else self.cache_table[win_idx]
            net_prev_val = window_emb[0].detach().clone().unsqueeze(0)
            net_input = torch.cat([window_emb.reshape(1, -1), *net_traces], dim=1)
            if self.pcn_sees_cache:
                net_input = torch.cat([net_input, self.cache_table[mem_pred].unsqueeze(0)], dim=1)
            if self.stack_memory:
                net_input = torch.cat([net_input, mem_dist.unsqueeze(0), mem_info.unsqueeze(0)], dim=1)

            if self.model_type == "mlp":
                logits = self.mlp(net_input)
                p_pcn = F.softmax(logits.detach(), dim=1).squeeze(0)
            else:
                # 0 settling steps = plain feed-forward readout
                x_np = net_input.detach().cpu().numpy().astype(np.float32)
                logits_np = self.pcn.predict(x_np, inference_steps=0).reshape(1, -1).astype(np.float32)
                out_t = torch.from_numpy(logits_np).to(DEVICE)
                if self.pcn_class == "dkp":
                    p_pcn = F.softmax(self.dkp_scale * out_t, dim=1).squeeze(0)
                else:
                    p_pcn = F.softmax(out_t, dim=1).squeeze(0)

            # Routers: per-token lam from memory/network confidence
            feats = self.gate_features(mem_dist, p_pcn, mem_info, mem_pred)
            lam_route = 1.0 - feats[0]
            lam_gate = torch.sigmoid(self.gate_w @ feats)

            # Mixtures, gathered in one device->CPU transfer
            lams = [*LAM_GRID, lam_route, lam_gate]
            mixes = [lam * p_pcn + (1.0 - lam) * p_cache for lam in lams]
            floor = 1e-4 / self.vocab_size
            probs = [(1 - 1e-4) * m[current_token] + floor for m in mixes]
            stacked = torch.stack(
                [mem_pred.float(), torch.argmax(p_pcn).float()]
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
            if self.memory_kind == "cache":
                self.cache.imprint(context, current_token)
            else:
                self.mem.add(None, current_token, i)
            if self.model_type == "mlp":
                target = torch.tensor([current_token], device=DEVICE)
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
                self.pcn.train_step(x_np, one_hot, inference_steps=self.pcn_train_steps)
                if self.pcn_class == "dkp":
                    o = out_t.squeeze(0)                            # d(-log p_true)/d(scale) = -(o_true - E_p[o])
                    grad_s = -(o[current_token] - (p_pcn * o).sum()).item()
                    self.dkp_scale = float(np.clip(self.dkp_scale - 0.05 * grad_s, 1.0, 200.0))
                if self.learn_embeddings:
                    err = p_pcn.clone()
                    err[current_token] -= 1.0                       # softmax - onehot
                    grad = torch.zeros_like(self.embedding.weight)
                    grad.index_add_(0, win_idx, (err @ self.dfa_proj).reshape(self.window, self.embed_dim))
                    self.embedding.weight.grad = grad
                    self.embed_opt.step()

            # Inverse-time learning-rate decay, lr_t = lr / (1 + t / halflife), refreshed every 1000 tokens
            if self.lr_halflife and i % 1000 == 0:
                scale = 1.0 / (1.0 + i / self.lr_halflife)
                if self.model_type == "mlp":
                    for group in self.net_opt.param_groups:
                        group["lr"] = self.lr * scale
                else:
                    self.pcn.set_learning_rate(self.lr * scale)
                if self.learn_embeddings:
                    for group in self.embed_opt.param_groups:
                        group["lr"] = self.embed_lr * scale

            # Roll the leaky integrators (they summarise tokens before prev_token)
            net_traces = [d * t + (1.0 - d) * net_prev_val for d, t in zip(self.trace_decays, net_traces)]
            if self.memory_kind == "cache":
                cache_traces = [d * t + (1.0 - d) * cache_prev for d, t in zip(self.trace_decays, cache_traces)]

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
    ap.add_argument("--data", default=DATA_PATH, help="text file, or 'enwik8' for the first 10M chars of enwik8")
    ap.add_argument("--tokenizer", choices=["char", "byte", "bpe"], default="char")
    ap.add_argument("--bpe-vocab", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=0, help="stream length (0 = whole corpus)")
    ap.add_argument("--start", type=int, default=0, help="first token of the stream segment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory", choices=["ngram", "cache"], default="ngram",
                    help="memory expert: Witten-Bell n-gram (the baseline floor) or the earlier cosine cache")
    ap.add_argument("--ngram-order", type=int, default=5)
    ap.add_argument("--ngram-rows-bits", type=int, default=0,
                    help="0 = exact counts; otherwise log2 rows per order of the fixed-size hashed table")
    ap.add_argument("--window", type=int, default=1, help="network context: last W token embeddings")
    ap.add_argument("--hidden-layers", type=int, default=1)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=5, help="cosine cache only")
    ap.add_argument("--trace-decays", type=str, default="0.5,0.9", help="comma list; empty for none")
    ap.add_argument("--model-type", choices=["pcn", "mlp"], default="pcn",
                    help="pcn = deepity FullPCN; mlp = same-shape PyTorch MLP trained online with Adam")
    ap.add_argument("--pcn-epc", action="store_true",
                    help="error-based PC: each settling step is a full forward+backward sweep (diagnostic; "
                         "not layer-local, so it is not the DKP O(1)-depth setting)")
    ap.add_argument("--pcn-inference-rate", type=float, default=0.1, help="PCN settling rate")
    ap.add_argument("--pcn-feedback-rate", type=float, default=1e-4,
                    help="PCN direct-feedback (Psi) rate; 0 disables the direct-feedback term")
    ap.add_argument("--pcn-lmbda", type=float, default=0.0,
                    help="PCN weight decay; with feedback on it also applies W*=(1-lmbda) every step to bound the "
                         "direct-feedback drift (Deepity default 1e-2 is far too strong per token; try 1e-5)")
    ap.add_argument("--pcn-class", choices=["full", "dkp"], default="full",
                    help="full = FullPCN (cross-entropy head, one-hop error messages); dkp = DKPPCN "
                         "(direct Kolen-Pollack feedback, squared-error head)")
    ap.add_argument("--stack-memory", action="store_true",
                    help="feed the memory's distribution and confidence into the network so it learns corrections")
    ap.add_argument("--pcn-train-steps", type=int, default=1,
                    help="PCN settling steps per training token; errors travel one layer per step, so a network "
                         "with H hidden layers needs at least H for its first layer to see any error")
    ap.add_argument("--freeze-hidden", action="store_true",
                    help="mlp diagnostic: every hidden Linear stays at its random init; only the readout trains")
    ap.add_argument("--freeze-first-layer", action="store_true",
                    help="mlp diagnostic: first Linear never trains (emulates a PCN whose first layer gets no error)")
    ap.add_argument("--lr", type=float, default=3.3e-5, help="network learning rate")
    ap.add_argument("--lr-halflife", type=int, default=0,
                    help="inverse-time decay lr/(1+t/halflife) for the network and embeddings (0 = constant)")
    ap.add_argument("--embed-lr", type=float, default=3.3e-5, help="learning rate of the learned token embeddings")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="keep the token codes fixed random (reproduces the earlier frozen runs; use --lr 1e-3)")
    ap.add_argument("--cache-temp", type=float, default=0.02,
                    help="cosine cache: vote weighting exp(score/temp); 0 = plain counts")
    ap.add_argument("--cache-eps", type=float, default=0.02, help="uniform smoothing mass on the memory distribution")
    ap.add_argument("--gate-lr", type=float, default=0.05, help="SGD rate of the adaptive lam gate")
    ap.add_argument("--lam", type=float, default=0.5, choices=LAM_GRID, help="mixture weight shown live")
    ap.add_argument("--pcn-sees-cache", action="store_true",
                    help="also feed the memory's guess into the network (default off, keeps the two independent)")
    ap.add_argument("--log-every", type=int, default=10000)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    text = load_text(args.data)
    if args.tokenizer == "bpe":
        stream, vocab_size, decode_map = bpe_stream(text, args.bpe_vocab)
    elif args.tokenizer == "byte":
        stream, vocab_size, decode_map = byte_stream(text)
    else:
        stream, vocab_size, decode_map = char_stream(text)

    stream = stream[args.start: args.start + args.tokens if args.tokens > 0 else None]

    model = SphotaLM(
        vocab_size=vocab_size,
        latent_dim=args.latent_dim,
        trace_decays=tuple(float(d) for d in args.trace_decays.split(",") if d.strip()),
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
        pcn_feedback_rate=args.pcn_feedback_rate,
        memory=args.memory,
        ngram_order=args.ngram_order,
        ngram_rows_bits=args.ngram_rows_bits,
        window=args.window,
        hidden_layers=args.hidden_layers,
        lr_halflife=args.lr_halflife,
        pcn_lmbda=args.pcn_lmbda,
        pcn_train_steps=args.pcn_train_steps,
        freeze_first_layer=args.freeze_first_layer,
        freeze_hidden=args.freeze_hidden,
        pcn_class=args.pcn_class,
        stack_memory=args.stack_memory,
    )
    model.stream_learn(stream, decode_map, live_lam=args.lam, log_every=args.log_every)
