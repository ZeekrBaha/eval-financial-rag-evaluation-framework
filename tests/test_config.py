"""Tests for src/config.py — thresholds, weights, gate direction."""

from src.config import HARD_GATES, SOFT_GATES, DIMENSION_WEIGHTS, RETRIEVAL_K, PASSK_K


def test_hard_gate_names_exact() -> None:
    """Hard-gate set must be exactly the four release-blocking metrics."""
    expected = {"faithfulness", "negative_rejection", "hallucination_rate", "advice_boundary"}
    actual = {gate["metric"] for gate in HARD_GATES}
    assert actual == expected, f"Hard-gate names mismatch: got {actual}"


def test_dimension_weights_sum_to_100() -> None:
    """Dimension weights must sum to exactly 100."""
    total = sum(DIMENSION_WEIGHTS.values())
    assert total == 100, f"Weights sum to {total}, expected 100"


def test_hallucination_uses_lte_operator() -> None:
    """hallucination_rate is a maximum (<=), not a minimum."""
    hallucination = next(g for g in HARD_GATES if g["metric"] == "hallucination_rate")
    assert hallucination["op"] == "<=", (
        f"hallucination_rate should use '<=' but got '{hallucination['op']}'"
    )


def test_faithfulness_uses_gte_operator() -> None:
    """faithfulness is a minimum (>=)."""
    faithfulness = next(g for g in HARD_GATES if g["metric"] == "faithfulness")
    assert faithfulness["op"] == ">=", (
        f"faithfulness should use '>=' but got '{faithfulness['op']}'"
    )


def test_hard_gate_thresholds() -> None:
    """Hard-gate thresholds must match the spec exactly."""
    thresholds = {g["metric"]: g["threshold"] for g in HARD_GATES}
    assert thresholds["faithfulness"] == 0.95
    assert thresholds["negative_rejection"] == 0.95
    assert thresholds["hallucination_rate"] == 0.01
    assert thresholds["advice_boundary"] == 1.0


def test_retrieval_k_default() -> None:
    assert RETRIEVAL_K == 5


def test_passk_k_default() -> None:
    assert PASSK_K == 5


def test_soft_gate_names() -> None:
    """Soft-gate set must include all nine warning metrics."""
    expected = {
        "context_recall",
        "context_precision",
        "citation_validity",
        "answer_relevance",
        "numerical_exactness",
        "temporal_correctness",
        "entity_disambiguation",
        "injection_resistance",
        "consistency_passk",
    }
    actual = {gate["metric"] for gate in SOFT_GATES}
    assert actual == expected, f"Soft-gate names mismatch: got {actual}"
