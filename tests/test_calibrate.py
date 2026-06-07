"""Tests for src/eval/calibrate.py — judge calibration via Cohen's κ (offline).

All offline: judge verdicts come from the recorded calibration fixture, so κ is
deterministic. No key, no network.
"""

from __future__ import annotations

import pytest

from src.config import KAPPA_TARGET
from src.eval.calibrate import (
    calibrate,
    cohen_kappa,
    load_calibration_set,
    main,
)


# ---------------------------------------------------------------------------
# cohen_kappa — the statistic itself
# ---------------------------------------------------------------------------


class TestCohenKappa:
    def test_perfect_agreement_is_one(self) -> None:
        assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)

    def test_constant_identical_raters_is_one(self) -> None:
        # p_e == 1 (both raters always 1) → defined as perfect agreement.
        assert cohen_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == pytest.approx(1.0)

    def test_total_disagreement_is_minus_one(self) -> None:
        assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)

    def test_known_partial_case(self) -> None:
        # 9/10 agreement with balanced-ish marginals → κ = 0.8 (hand-computed).
        ref = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
        judge = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
        assert cohen_kappa(ref, judge) == pytest.approx(0.8, abs=1e-9)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cohen_kappa([1, 0], [1, 0, 1])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            cohen_kappa([], [])


# ---------------------------------------------------------------------------
# Calibration set loading
# ---------------------------------------------------------------------------


class TestLoadCalibrationSet:
    def test_loads_all_rows(self) -> None:
        items = load_calibration_set()
        assert len(items) == 10

    def test_set_is_balanced(self) -> None:
        """κ needs label variance — the fixture must contain both pass and fail."""
        items = load_calibration_set()
        hallucinated = sum(i.ref_hallucinated for i in items)
        assert 0 < hallucinated < len(items)
        faithful = sum(1 for i in items if i.ref_faithful)
        assert 0 < faithful < len(items)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_calibration_set("datasets/does_not_exist.jsonl")


# ---------------------------------------------------------------------------
# End-to-end offline calibration
# ---------------------------------------------------------------------------


class TestCalibrateOffline:
    def test_reports_both_dimensions(self) -> None:
        report = calibrate()
        assert report["n"] == 10
        assert "faithfulness" in report
        assert "hallucination" in report

    def test_kappa_values_are_deterministic(self) -> None:
        report = calibrate()
        faith = report["faithfulness"]
        halluc = report["hallucination"]
        assert isinstance(faith, dict) and isinstance(halluc, dict)
        assert faith["kappa"] == pytest.approx(0.8, abs=1e-6)
        assert halluc["kappa"] == pytest.approx(0.7826, abs=1e-3)

    def test_status_reflects_target(self) -> None:
        report = calibrate()
        faith = report["faithfulness"]
        halluc = report["hallucination"]
        assert isinstance(faith, dict) and isinstance(halluc, dict)
        # Both κ ≥ KAPPA_TARGET in the shipped fixture.
        assert faith["kappa"] >= KAPPA_TARGET
        assert halluc["kappa"] >= KAPPA_TARGET
        assert faith["status"] == "CALIBRATED"
        assert halluc["status"] == "CALIBRATED"

    def test_main_exits_zero_when_calibrated(self) -> None:
        assert main([]) == 0
