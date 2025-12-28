import numpy as np
import pytest

from metrics import calculate_cohens_kappa
from tests.test_metrics.constants import ONLY_NEGATIVES, ONLY_POSITIVES, PARTIAL_1, PARTIAL_2, \
    PERFECT_MATCH, \
    ZERO_KAPPA_1, ZERO_KAPPA_2


class TestKappaScore:
    def test_kappa_partial_match(self):
        # TP = 1, FN = 1, FP = 0, TN = 2 -> p_o = 3/4, p_e = 1/2 -> k = 0.5
        result = calculate_cohens_kappa(PARTIAL_1, PARTIAL_2)
        assert result == 0.5

    def test_kappa_perfect_match(self):
        assert calculate_cohens_kappa(PERFECT_MATCH, PERFECT_MATCH) == 1.0

    def test_kappa_perfect_match_only_positives(self):
        assert calculate_cohens_kappa(ONLY_POSITIVES, ONLY_POSITIVES) == 1.0

    def test_kappa_zero_division_handling(self):
        assert calculate_cohens_kappa(ONLY_NEGATIVES, ONLY_NEGATIVES) == 1.0

    def test_kappa_complete_mismatch(self):
        assert calculate_cohens_kappa(ONLY_POSITIVES, ONLY_NEGATIVES) == 0.0

    def test_kappa_score_0(self):
        result = calculate_cohens_kappa(ZERO_KAPPA_1, ZERO_KAPPA_2)
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
