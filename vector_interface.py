"""Vector interface for externally managed token vectors.

This module does not create, initialize, serialize, or persist vectors.
It forwards calls to an externally supplied provider.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class VectorProvider(Protocol):
    def get_vector(self, token_id: int) -> Sequence[float]:
        ...

    def update_vectors(self, token_ids: list[int], delta: float) -> None:
        ...


_provider: VectorProvider | None = None


def register_provider(provider: VectorProvider) -> None:
    global _provider
    _provider = provider


def get_vector(token_id: int) -> Sequence[float]:
    if _provider is None:
        raise RuntimeError("No external vector provider registered")
    return _provider.get_vector(token_id)


def update_vectors(token_ids: list[int], delta: float) -> None:
    if _provider is None:
        raise RuntimeError("No external vector provider registered")
    _provider.update_vectors(token_ids, delta)
