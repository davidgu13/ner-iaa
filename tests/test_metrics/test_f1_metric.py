import numpy as np
import pytest

from metrics import calculate_f1
from tests.test_metrics.constants import ONLY_NEGATIVES, ONLY_POSITIVES, PARTIAL_1, PARTIAL_2, \
    PERFECT_MATCH


class TestF1Score:
    def test_f1_partial_match(self):
        # Precision = 1.0, Recall = 0.5 -> F1 = 2 * (1 * 0.5) / (1 + 0.5) = 2 / 3
        result = calculate_f1(PARTIAL_1, PARTIAL_2)
        assert result == pytest.approx(0.6666, rel=1e-3)

    def test_f1_perfect_match(self):
        assert calculate_f1(PERFECT_MATCH, PERFECT_MATCH) == 1.0

    def test_f1_perfect_match_only_positives(self):
        assert calculate_f1(ONLY_POSITIVES, ONLY_POSITIVES) == 1.0

    def test_f1_zero_division_handling(self):
        """
        aka perfect_match_only_negatives.
        Verifies that zero_division=1 prevents exceptions when
        no positive labels are predicted or present.
        """
        # Both sequences are all zeros; F1 is undefined but should return 1.0
        assert calculate_f1(ONLY_NEGATIVES, ONLY_NEGATIVES) == 1.0

    def test_f1_complete_mismatch(self):
        assert calculate_f1(ONLY_POSITIVES, ONLY_NEGATIVES) == 0.0

    def test_f1_different_lengths(self):
        with pytest.raises(ValueError):
            calculate_f1([1, 1], [1, 0, 1])

    def test_f1_empty_list(self):
        with pytest.raises(ValueError):
            calculate_f1([], [1, 0, 1])

    def test_f1_both_lists_empty(self):
        assert calculate_f1([], []) == pytest.approx(np.nan, nan_ok=True)
