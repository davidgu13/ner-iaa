import numpy as np
import pytest

from metrics import calculate_cohens_kappa, calculate_f1

"""
TODO:
1. Separate to 2 classes
2. Extract the cases to constants
3. Test also for "O"
"""


class TestMetrics:
    ## Tests for calculate_f1
    def test_f1_partial_match(self):
        # Precision = 1.0, Recall = 0.5 -> F1 = 2 * (1 * 0.5) / (1 + 0.5) = 2 / 3
        result = calculate_f1([1, 1, 0, 0], [1, 0, 0, 0])
        assert result == pytest.approx(0.6666, rel=1e-3)

    def test_f1_perfect_match(self):
        assert calculate_f1([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0

    def test_f1_perfect_match_only_positives(self):
        assert calculate_f1([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0

    def test_f1_zero_division_handling(self):
        """
        aka perfect_match_only_negatives.
        Verifies that zero_division=1 prevents exceptions when
        no positive labels are predicted or present.
        """
        # Both sequences are all zeros; F1 is undefined but should return 1.0
        assert calculate_f1([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0

    def test_f1_complete_mismatch(self):
        assert calculate_f1([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0

    def test_f1_different_lengths(self):
        with pytest.raises(ValueError):
            calculate_f1([1, 1], [1, 0, 1])

    def test_f1_empty_list(self):
        with pytest.raises(ValueError):
            calculate_f1([], [1, 0, 1])

    def test_f1_both_lists_empty(self):
        assert calculate_f1([], []) == pytest.approx(np.nan, nan_ok=True)

    ## Tests for calculate_cohens_kappa
    def test_kappa_partial_match(self):
        # TP = 1, FN = 1, FP = 0, TN = 2 -> p_o = 3/4, p_e = 1/2 -> k = 0.5
        result = calculate_cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0])
        assert result == 0.5

    def test_kappa_perfect_match(self):
        result = calculate_cohens_kappa([1, 0, 1, 1], [1, 0, 1, 1])
        assert result == 1.0

    def test_kappa_perfect_match_only_positives(self):
        assert calculate_cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0

    def test_kappa_zero_division_handling(self):
        assert calculate_f1([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0

    def test_kappa_complete_mismatch(self):
        assert calculate_cohens_kappa([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0

    def test_kappa_score_0(self):
        s1 = [1, 1, 0, 0]
        s2 = [1, 0, 0, 1]
        result = calculate_cohens_kappa(s1, s2)
        # Observed = 0.5, Chance = 0.5 -> Kappa = (observed - chance) / (1 - chance) = 0.0
        assert result == 0.0

    def test_kappa_different_lengths(self):
        with pytest.raises(ValueError):
            calculate_cohens_kappa([1, 1], [1, 0, 1])

    def test_kappa_empty_list(self):
        with pytest.raises(ValueError):
            calculate_cohens_kappa([], [1, 0, 1])

    def test_kappa_both_lists_empty(self):
        assert calculate_cohens_kappa([], []) == pytest.approx(np.nan, nan_ok=True)
