"""Unit tests for Reciprocal Rank Fusion."""

from retrieval.hybrid.rrf_fusion import reciprocal_rank_fusion


def test_rrf_merges_two_lists():
    dense = [
        {"chunk_id": "a", "content": "...", "score": 0.9},
        {"chunk_id": "b", "content": "...", "score": 0.7},
        {"chunk_id": "c", "content": "...", "score": 0.5},
    ]
    sparse = [
        {"chunk_id": "b", "content": "...", "score": 0.8},
        {"chunk_id": "a", "content": "...", "score": 0.6},
        {"chunk_id": "d", "content": "...", "score": 0.4},
    ]
    result = reciprocal_rank_fusion([dense, sparse])
    ids = [r["chunk_id"] for r in result]

    # 'b' ranked 2nd in dense and 1st in sparse → should rank high
    # 'a' ranked 1st in dense and 2nd in sparse → should rank high
    assert "a" in ids
    assert "b" in ids
    assert "d" in ids  # Only in sparse but still included


def test_rrf_scores_descending():
    dense = [{"chunk_id": str(i), "content": "", "score": 1.0 / (i + 1)} for i in range(5)]
    result = reciprocal_rank_fusion([dense])
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_single_list():
    items = [{"chunk_id": "x", "content": "test", "score": 0.8}]
    result = reciprocal_rank_fusion([items])
    assert len(result) == 1
    assert result[0]["chunk_id"] == "x"


def test_rrf_empty_list():
    result = reciprocal_rank_fusion([[], []])
    assert result == []
