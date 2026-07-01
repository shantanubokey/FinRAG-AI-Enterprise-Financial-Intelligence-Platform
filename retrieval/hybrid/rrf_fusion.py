"""
Reciprocal Rank Fusion (RRF) for hybrid search.

RRF formula: score(d) = sum(1 / (k + rank(d)))
where k=60 is a smoothing constant (empirically best from the original paper).

Why RRF over weighted sum:
- No need to tune weights for each query type
- Robust to score scale differences between dense and sparse
- Consistently outperforms weighted sum in BEIR benchmarks
"""

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Merge multiple ranked lists using RRF.

    Args:
        result_lists: Each list is a ranked set of results with 'chunk_id' key.
        k: RRF smoothing constant (default 60 from original paper).
        weights: Optional per-list weights. Defaults to equal weighting.

    Returns:
        Single merged list sorted by RRF score, highest first.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError("weights length must match result_lists length")

    # chunk_id → accumulated RRF score
    scores: dict[str, float] = defaultdict(float)
    # chunk_id → payload (keep the one with highest original score)
    payloads: dict[str, dict[str, Any]] = {}

    for result_list, weight in zip(result_lists, weights):
        for rank, result in enumerate(result_list, start=1):
            chunk_id = result.get("chunk_id") or result.get("id", str(rank))
            rrf_score = weight * (1.0 / (k + rank))
            scores[chunk_id] += rrf_score

            # Keep payload; prefer higher original score
            if chunk_id not in payloads or result.get("score", 0) > payloads[chunk_id].get("score", 0):
                payloads[chunk_id] = result

    # Sort by final RRF score
    merged = []
    for chunk_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(payloads[chunk_id])
        item["rrf_score"] = round(rrf_score, 6)
        item["original_score"] = item.pop("score", 0.0)
        item["score"] = rrf_score
        merged.append(item)

    return merged
