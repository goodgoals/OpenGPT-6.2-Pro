"""Survival filter based only on internal vector coherence."""

from __future__ import annotations

import math
from typing import Sequence

from vector_interface import get_vector


def _dot(v1: Sequence[float], v2: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return _dot(v1, v2) / (n1 * n2)


def coherence_score(token_sequence: list[int]) -> float:
    if len(token_sequence) < 2:
        return 0.0
    sims = []
    for left, right in zip(token_sequence, token_sequence[1:]):
        sims.append(_cosine_similarity(get_vector(left), get_vector(right)))
    return sum(sims) / len(sims) if sims else 0.0


def survives(token_sequence: list[int], threshold: float = 0.10) -> tuple[bool, float]:
    score = coherence_score(token_sequence)
    return score >= threshold, score
