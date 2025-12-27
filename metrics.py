import pandas as pd
from irrCAC.table import CAC
from sklearn.metrics import f1_score


def calculate_f1(sequence1: list, sequence2: list) -> float:
    return f1_score(sequence1, sequence2, zero_division=0)  # TODO think if zero_division has the correct value


def calculate_cohens_kappa(sequence1: list, sequence2: list) -> float:
    crosstab = pd.crosstab(sequence1, sequence2)
    return CAC(crosstab).cohen()['est']['coefficient_value']
