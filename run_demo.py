"""End-to-end demo for emergent structure via survival pressure."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass

from survival_filter import survives
from trainer import generate_sequence, run_training
from vector_interface import register_provider


TOKENS_CSV = "tokens.csv"


@dataclass(frozen=True)
class Token:
    token_id: int
    token_string: str
    token_type: str


class ExternalVectorBackend:
    """Demo-only stand-in for an external vector service."""

    def __init__(self, token_ids: list[int], dim: int = 12, seed: int = 42):
        rng = random.Random(seed)
        self._vectors = {
            tid: [rng.uniform(-1.0, 1.0) for _ in range(dim)]
            for tid in token_ids
        }

    def get_vector(self, token_id: int) -> list[float]:
        return self._vectors[token_id]

    def update_vectors(self, token_ids: list[int], delta: float) -> None:
        if len(token_ids) != 2:
            return
        a, b = token_ids
        va = self._vectors[a]
        vb = self._vectors[b]
        midpoint = [(x + y) * 0.5 for x, y in zip(va, vb)]
        self._vectors[a] = [x + delta * 0.2 * (m - x) for x, m in zip(va, midpoint)]
        self._vectors[b] = [y + delta * 0.2 * (m - y) for y, m in zip(vb, midpoint)]


def load_tokens(path: str) -> list[Token]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Token(int(r["token_id"]), r["token_string"], r["token_type"]) for r in csv.DictReader(f)]


def render(sequence: list[int], id_to_str: dict[int, str]) -> str:
    return " ".join(id_to_str[i] for i in sequence)


def parse_input(text: str, str_to_id: dict[str, int]) -> list[int]:
    # Extremely plain tokenization: lowercase split + punctuation isolation.
    flat = text.lower().replace(".", " . ").replace(",", " , ").split()
    return [str_to_id.get(tok, str_to_id["<UNK>"]) for tok in flat]


def main() -> None:
    tokens = load_tokens(TOKENS_CSV)
    token_ids = [t.token_id for t in tokens]
    id_to_str = {t.token_id: t.token_string for t in tokens}
    str_to_id = {t.token_string: t.token_id for t in tokens}

    backend = ExternalVectorBackend(token_ids)
    register_provider(backend)

    user_input = "Jake started the car."
    encoded_input = parse_input(user_input, str_to_id)
    print(f"Input: {user_input}")
    print("Encoded input:", encoded_input)

    print("\nBefore training:")
    pre_rng = random.Random(3)
    for i in range(3):
        seq = generate_sequence(token_ids, max_len=8, rng=pre_rng)
        ok, score = survives(seq)
        print(f"  [{i}] {render(seq, id_to_str)}")
        print(f"      coherence={score:.3f} survive={ok}")

    run_training(token_ids, steps=250, max_len=8, rng_seed=11)

    print("\nAfter training:")
    post_rng = random.Random(3)
    for i in range(5):
        seq = generate_sequence(token_ids, max_len=8, rng=post_rng)
        ok, score = survives(seq)
        print(f"  [{i}] {render(seq, id_to_str)}")
        print(f"      coherence={score:.3f} survive={ok}")


if __name__ == "__main__":
    main()
