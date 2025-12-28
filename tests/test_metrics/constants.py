# Binary Labels Examples
PARTIAL_1, PARTIAL_2 = [1, 1, 0, 0], [1, 0, 0, 0]
PERFECT_MATCH = [1, 0, 1, 1]
ONLY_POSITIVES, ONLY_NEGATIVES = [1, 1, 1, 1], [0, 0, 0, 0]
ZERO_KAPPA_1, ZERO_KAPPA_2 = [1, 1, 0, 0], [1, 0, 0, 1]

# NER Labels Examples
PARTIAL_LOC_1, PARTIAL_LOC_2 = ["LOC", "LOC", "O", "O"], ["LOC", "O", "O", "O"]
PERFECT_MATCH_LOC = ["LOC", "O", "LOC", "LOC"]
ONLY_LOC, ONLY_O = ["LOC", "LOC", "LOC", "LOC"], ["O", "O", "O", "O"]
ZERO_KAPPA_STR_1, ZERO_KAPPA_STR_2 = ["LOC", "LOC", "O", "O"], ["LOC", "O", "O", "LOC"]
