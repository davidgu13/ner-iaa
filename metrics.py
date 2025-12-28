import numpy as np
import pandas as pd
from irrCAC.table import CAC
from sklearn.metrics import f1_score


def calculate_f1(sequence1: list, sequence2: list) -> float:
    if not sequence1 and not sequence2:
        return np.nan

    return f1_score(sequence1, sequence2, zero_division=1)


def calculate_cohens_kappa(sequence1: list, sequence2: list) -> float:
    """
        Calculates Cohen's Kappa coefficient for two sequences, handling edge cases and
        ensuring a complete contingency table.

        The function manually handles cases of zero variance (unanimous agreement
        on a single class) which typically cause division-by-zero errors in
        standard Kappa implementations.

        Edge Cases:
        1. If both sequences are empty, returns np.nan.
        2. If both sequences contain only "O" or 0, returns 1.0.
        3. If both sequences contain only non-"O"/non-0 labels (unanimous
           agreement on a specific entity), returns 1.0.
        4. In mathematical scenarios where agreement is purely due to chance
           or the divisor is zero, the underlying irrCAC library may return np.nan.

        Args:
            sequence1 (list): The labels assigned by the first rater.
            sequence2 (list): The labels assigned by the second rater.

        Returns:
            float: Cohen's Kappa coefficient or np.nan if the calculation is undefined.
    """
    if not sequence1 and not sequence2:
        return np.nan

    set1, set2 = set(sequence1), set(sequence2)
    unique_labels = set1 | set2

    if unique_labels == {"O"} or unique_labels == {0}:
        return 1.0

    if "O" not in unique_labels and 0 not in unique_labels:
        return 1.0

    # 2. Convert to Categorical with the full set of categories
    # This ensures both series have the same "vocabulary"
    s1 = pd.Categorical(sequence1, categories=unique_labels)
    s2 = pd.Categorical(sequence2, categories=unique_labels)

    # 3. Use dropna=False to keep rows/cols with 0 counts
    crosstab = pd.crosstab(s1, s2, dropna=False)
    return CAC(crosstab).cohen()['est']['coefficient_value']
