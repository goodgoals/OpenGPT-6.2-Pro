"""Training through survival pressure only.

No backpropagation through token prediction and no global objective.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from probability_logic import sample_next_token
from survival_filter import survives
from vector_interface import update_vectors


@dataclass
class EpisodeResult:
    sequence: list[int]
    survived: bool
    coherence: float


def generate_sequence(token_ids: list[int], max_len: int, rng: random.Random) -> list[int]:
    seq: list[int] = [0]  # <BOS>
    while len(seq) < max_len:
        nxt = sample_next_token(token_ids, seq, rng=rng)
        seq.append(nxt)
        if nxt == 1:  # <EOS>
            break
    if seq[-1] != 1:
        seq.append(1)
    return seq


def apply_local_updates(
    sequence: list[int],
    survived_flag: bool,
    step_size: float = 0.005,
    weaken_failures: bool = False,
) -> None:
    if survived_flag:
        signed = step_size
    elif weaken_failures:
        signed = -step_size * 0.25
    else:
        return

    for left, right in zip(sequence, sequence[1:]):
        update_vectors([left, right], signed)


def run_training(
    token_ids: list[int],
    steps: int,
    max_len: int = 8,
    rng_seed: int = 7,
) -> list[EpisodeResult]:
    rng = random.Random(rng_seed)
    history: list[EpisodeResult] = []
    for _ in range(steps):
        seq = generate_sequence(token_ids, max_len=max_len, rng=rng)
        ok, score = survives(seq)
        apply_local_updates(seq, ok)
        history.append(EpisodeResult(sequence=seq, survived=ok, coherence=score))
    return history
