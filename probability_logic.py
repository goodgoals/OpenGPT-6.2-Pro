"""Deliberately weak next-token generator.

This module contains no trainable parameters and no persistent state.
"""

from __future__ import annotations

import random
from typing import Iterable


def sample_next_token(token_ids: Iterable[int], context: list[int], rng: random.Random | None = None) -> int:
    """Return one uniformly sampled token id with tiny rule-based filtering.

    Rules are intentionally minimal:
    - position 0 cannot be <EOS>
    - token <BOS> cannot appear after position 0
    """
    local_rng = rng or random
    ids = list(token_ids)

    if not ids:
        raise ValueError("token_ids cannot be empty")

    filtered = []
    for token_id in ids:
        if len(context) == 0 and token_id == 1:  # avoid immediate <EOS>
            continue
        if len(context) > 0 and token_id == 0:  # avoid nested <BOS>
            continue
        filtered.append(token_id)

    if not filtered:
        raise ValueError("Filtering removed all available token ids")

    return local_rng.choice(filtered)
