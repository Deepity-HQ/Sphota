"""Sphota: Continuous Learning PCN Model with Associative Fast-Cache."""

from pydeepity.layer import Linear, ReLU
import pydeepity as pd

import numpy as np

import torch
import torch.nn as nn

class FastCache:
    """Stores latent states to predict the next token"""
    def __init__(self, latent_dim: int):
        self.keys = torch.zeros(0, latent_dim).cuda()
        self.next_tokens: list[int] = []

    def retrieve(self, latent_state: torch.Tensor, k: int = 1) -> int | None:
        """Returns the token(s) that followed the most similar past states."""
        if len(self.keys) == 0:
            return None

        scores: np.ndarray = self.keys @ latent_state.T
        best_idx = int(torch.argmax(scores))
        return self.next_tokens[best_idx]

    def imprint(self, latent_state: np.ndarray, actual_next_token: str) -> None:
        """Saves the current state and what token came next."""
        self.keys = torch.cat([self.keys, latent_state])
        self.next_tokens.append(actual_next_token)

class SphotaLM:
    def __init__(self, vocab_size: int, embed_dim: int = 128, latent_dim: int = 256):
        self.embedding = nn.Embedding(vocab_size, embed_dim).cuda()

        self.pcn = pd.DKPPCN(
            Linear(embed_dim, 256),
            ReLU(),
            Linear(256, latent_dim),
            batch_size=1,
            device="cuda"
        )

        self.pcn.configure(
            learning_rate=1e-3,
            inference_rate=0.1,
            feedback_rate=1e-4,
            lmbda=0.0,
            optimizer="ADAMW"
        )

        self.cache = FastCache(latent_dim)

    def stream_learn(self, token_stream: list[str]):
        """Processes a continuous 1D stream of text tokens."""
        print("[Info] Starting continuous learning stream")

        prev_token = token_stream[0]

        for i in range(1, len(token_stream)):
            current_token = token_stream[i]

            x_emb = self.embedding(torch.tensor([prev_token]).cuda()).detach().cpu().numpy().astype(np.float32)
            y_emb = self.embedding(torch.tensor([current_token]).cuda()).detach().cpu().numpy().astype(np.float32)

            self.pcn.predict(x_emb, inference_steps = 1)

            latent_state_np = np.array(self.pcn[-1].beliefs).reshape(1, -1)
            latent_state = torch.from_numpy(latent_state_np).cuda()

            predicted_token = self.cache_retrieve(latent_state)
            self.cache.imprint(latent_state, current_token)

            self.pcn.train_step(x_emb, y_emb, inference_steps=1)

            if i % 1000 == 0:
                acc_msg: str = f"Predicted: {predicted_token} | Actual: {current_token}"
                print(f"Token {i} | Cache Size: {len(self.cache.keys)} | {acc_msg}")

            prev_token = current_token


if __name__ == "__main__":
    dummy_stream = np.random.randint(0, 1000, size=5000)
    model = SphotaLM(vocab_size=1000)
    model.stream_learn(dummy_stream)