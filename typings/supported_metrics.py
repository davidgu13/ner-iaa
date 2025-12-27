from enum import Enum


class SupportedMetrics(Enum):
    f1 = "f1"
    cohens_kappa = "cohens_kappa"
