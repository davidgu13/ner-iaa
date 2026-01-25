from typings.ner_label import NERLabel
from typings.word_span import WordSpan

# Real-world example with space-level tokenization:
REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT = "Paris Whitney Hilton , born February 17, 1981 is an American television " \
                                        "personality and businesswoman . She is the great-granddaughter of " \
                                        "Conrad Hilton , the founder of Hilton Hotels . Born in New York City and " \
                                        "raised in both California and New York , Hilton began a modeling career " \
                                        "when she signed with Donald Trump ’s modeling agency ."

# Ground Truth labels:
REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS1 = [[0, 20, 'PER'],  # Paris Whitney Hilton
                                                    [28, 45, 'TEMP'],  # February 17 , 1981
                                                    [138, 151, 'PER'],  # Conrad Hilton
                                                    [169, 182, 'ORG'],  # Hilton Hotels
                                                    [193, 206, 'LOC'],  # New York City
                                                    [226, 236, 'LOC'],  # California
                                                    [241, 249, 'LOC'],  # New York
                                                    [252, 258, 'PER'],  # Hilton
                                                    # [304, 316, 'PER'],    # Donald Trump  # Commented bc Nested NER isn't supported yet
                                                    [304, 335, 'ORG']]  # Donald Trump ’s modeling agency

# Annotatoed labels:
REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS2 = [[0, 20, 'PER'],  # Paris Whitney Hilton
                                                    [28, 45, 'TEMP'],  # February 17 , 1981
                                                    [138, 151, 'PER'],  # Conrad Hilton
                                                    [169, 182, 'ORG'],  # Hilton Hotels
                                                    [185, 189, 'PER'],  # Born
                                                    [193, 206, 'LOC'],  # New York City
                                                    [226, 236, 'LOC'],  # California
                                                    [241, 249, 'LOC'],  # New York
                                                    [252, 258, 'PER'],  # Hilton
                                                    [267, 275, 'LOC']]  # modeling

REAL_EXAMPLE_SIMPLE_TOKENIZATION_EXPECTED_SCORES_WITHOUT_O = {'PER': {'f1_score': 0.9231, 'cohens_kappa_score': 0.8947},
                                                              'LOC': {'f1_score': 0.9231, 'cohens_kappa_score': 0.8947},
                                                              'ORG': {'f1_score': 0.4444, 'cohens_kappa_score': 0.3617},
                                                              'TEMP': {'f1_score': 1.0, 'cohens_kappa_score': 1.0}}

REAL_EXAMPLE_SIMPLE_TOKENIZATION_EXPECTED_SCORES_WITH_O = {'PER': {'f1_score': 0.9231, 'cohens_kappa_score': 0.9136},
                                                           'LOC': {'f1_score': 0.9231, 'cohens_kappa_score': 0.9136},
                                                           'ORG': {'f1_score': 0.4444, 'cohens_kappa_score': 0.4135},
                                                           'TEMP': {'f1_score': 1.0, 'cohens_kappa_score': 1.0}}

# Real-world example with character-level evaluation:
REAL_EXAMPLE_TEXT = "Paris Whitney Hilton, born February 17, 1981 is an American television " \
                    "personality and businesswoman. She is the great-granddaughter of " \
                    "Conrad Hilton, the founder of Hilton Hotels. Born in New York City and " \
                    "raised in both California and New York, Hilton began a modeling career " \
                    "when she signed with Donald Trump’s modeling agency."

# Ground Truth labels:
REAL_EXAMPLE_DOCCANO_LABELS1 = [[0, 19, 'PER'],  # Paris Whitney Hilton
                                [27, 43, 'TEMP'],  # February 17 , 1981
                                [136, 148, 'PER'],  # Conrad Hilton
                                [166, 178, 'ORG'],  # Hilton Hotels
                                [189, 201, 'LOC'],  # New York City
                                [222, 231, 'LOC'],  # California
                                [237, 244, 'LOC'],  # New York
                                [247, 252, 'PER'],  # Hilton
                                # [299, 310, 'PER'],    # Donald Trump  # Commented bc Nested NER isn't supported yet
                                [299, 328, 'ORG']]  # Donald Trump’s modeling agency

# Annotatoed labels:
REAL_EXAMPLE_DOCCANO_LABELS2 = [[0, 20, 'PER'],  # Paris Whitney Hilton
                                [28, 45, 'TEMP'],  # February 17 , 1981
                                [138, 151, 'PER'],  # Conrad Hilton
                                [169, 182, 'ORG'],  # Hilton Hotels
                                [185, 189, 'PER'],  # Born
                                [193, 206, 'LOC'],  # New York City
                                [226, 236, 'LOC'],  # California
                                [241, 249, 'LOC'],  # New York
                                [252, 258, 'PER'],  # Hilton
                                [267, 275, 'LOC']]  # modeling

REAL_EXAMPLE_EXPECTED_SCORES_WITHOUT_O = {'PER': {'f1_score': 0.9231, 'cohens_kappa_score': 0.8947},
                                          'LOC': {'f1_score': 0.9231, 'cohens_kappa_score': 0.8947},
                                          'ORG': {'f1_score': 0.4444, 'cohens_kappa_score': 0.3617},
                                          'TEMP': {'f1_score': 1.0, 'cohens_kappa_score': 1.0}}

REAL_EXAMPLE_EXPECTED_SCORES_WITH_O = {'PER': {'f1_score': 0.9231, 'cohens_kappa_score': 0.9136},
                                       'LOC': {'f1_score': 0.9231, 'cohens_kappa_score': 0.9136},
                                       'ORG': {'f1_score': 0.4444, 'cohens_kappa_score': 0.4135},
                                       'TEMP': {'f1_score': 1.0, 'cohens_kappa_score': 1.0}}

# Dummy test cases:
PARSED_DOCCANO_LABELS1 = NERLabel.from_doccano_format_multiple_labels(REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS1)
PARSED_DOCCANO_LABELS2 = NERLabel.from_doccano_format_multiple_labels(REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS2)

TEXT_TO_WORD_SPANS_CASES = [
    ("Hello world", [
        WordSpan(text="Hello", start_index=0, end_index=5),
        WordSpan(text="world", start_index=6, end_index=11)
    ]),  # Standard case
    ("Word1    Word2\tWord3", [
        WordSpan(start_index=0, end_index=5, text="Word1"),
        WordSpan(start_index=9, end_index=14, text="Word2"),
        WordSpan(start_index=15, end_index=20, text="Word3")
    ]),  # Standard case
    ("  Trim   me  ", [
        WordSpan(text="Trim", start_index=2, end_index=6),
        WordSpan(text="me", start_index=9, end_index=11)
    ]),  # Multiple spaces and padding
    ("Price: $100!", [
        WordSpan(text="Price:", start_index=0, end_index=6),
        WordSpan(text="$100!", start_index=7, end_index=12)
    ]),  # Punctuation and special characters
    ("", []),  # Edge Case: Empty string
    ("  \n  \t  ", [])  # Edge Case: Only whitespace (tabs/newlines)
]

LABELS_TO_SEQUENCE_CASES = [
    # 1. Standard single-word entities
    ("John lives in London",
     NERLabel.from_doccano_format_multiple_labels([
         [0, 4, "PER"],
         [14, 20, "LOC"]
     ]),
     ["PER", "O", "O", "LOC"]
     ),
    # 2. Multi-word entity (Check if it handles sequence of tags)
    ("New York is cold",
     NERLabel.from_doccano_format_multiple_labels([[0, 8, "LOC"]]),
     ["LOC", "LOC", "O", "O"]
     ),
    # 3. No entities (All "O")
    (
        "Hello world",
        [],
        ["O", "O"]
    ),
    # 4. Entities with extra whitespace in text
    (
        "  Apple   Inc  ",
        NERLabel.from_doccano_format_multiple_labels([[2, 13, "ORG"]]),
        ["ORG", "ORG"]
    ),
    # 5. Real-world English text with simplified space-delimited tokenization - GT
    (
        REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT,
        PARSED_DOCCANO_LABELS1,
        ['PER', 'PER', 'PER', 'O', 'O', 'TEMP', 'TEMP', 'TEMP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O', 'O', 'PER', 'PER', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'O', 'O', 'O', 'LOC', 'LOC', 'LOC', 'O', 'O', 'O',
         'O', 'LOC', 'O', 'LOC', 'LOC', 'O', 'PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'ORG', 'ORG',
         'ORG', 'O']
    ),
    # 6. Real-world English text with simplified space-delimited tokenization - GT
    (
        REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT,
        PARSED_DOCCANO_LABELS2,
        ['PER', 'PER', 'PER', 'O', 'O', 'TEMP', 'TEMP', 'TEMP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O', 'O', 'PER', 'PER', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'O', 'PER', 'O', 'LOC', 'LOC', 'LOC', 'O', 'O', 'O',
         'O', 'LOC', 'O', 'LOC', 'LOC', 'O', 'PER', 'O', 'O', 'LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O']
    )
]

FILTER_NON_O_LABELS_CASES = [
    # 1. Standard filtering: Remove cases where both are "O"
    (
        ["O", "PER", "O", "LOC", "O"],
        ["O", "PER", "O", "O", "ORG"],
        ["PER", "LOC", "O"],
        ["PER", "O", "ORG"]
    ),
    # 2. Keep where only seq1 is non-O
    (
        ["ORG", "O"],
        ["O", "O"],
        ["ORG"],
        ["O"]
    ),
    # 2. Real example, filter out "O"
    (
        ["O", "O", "PER", "O", "O", "O", "PER", "PER", "O", "O", "ORG", "O"],
        ["O", "O", "PER", "PER", "O", "O", "PER", "PER", "O", "O", "O", "O"],
        ['PER', 'O', 'PER', 'PER', 'ORG'],
        ['PER', 'PER', 'PER', 'PER', 'O']
    ),
    # 4. All "O" results in empty lists
    (
        ["O", "O", "O"],
        ["O", "O", "O"],
        [],
        []
    ),
    # 5. No "O" at all (entire sequences preserved)
    (
        ["PER", "LOC"],
        ["ORG", "PER"],
        ["PER", "LOC"],
        ["ORG", "PER"]
    ),
    # 6. Empty input lists
    (
        [],
        [],
        [],
        []
    )
]

MASK_SEQUENCE_CASES = [
    # 1. ignore_o=False: Simple mapping of the target entity
    (
        False,
        "PER",
        ["PER", "O", "LOC"],
        ["O", "O", "PER"],
        [1, 0, 0],
        [0, 0, 1]  # seq2 has PER, but at index 2 (which is 0 in mask 1)
    ),
    # 2. ignore_o=True: 'O'-'O' pairs are dropped before masking
    # Input: ("PER", "O"), ("O", "O"), ("LOC", "PER")
    # Dropped: Index 1 ("O", "O")
    # Remaining: ("PER", "O"), ("LOC", "PER")
    (
        True,
        "PER",
        ["PER", "O", "LOC"],
        ["O", "O", "PER"],
        [1, 0],
        [0, 1]
    ),
    # 3. ignore_o=True: Target entity is "LOC"
    (
        True,
        "LOC",
        ["PER", "O", "LOC"],
        ["O", "O", "PER"],
        [0, 1],
        [0, 0]
    ),
    # 4. No occurrences of the entity
    (
        False, "ORG",
        ["PER", "LOC"],
        ["PER", "O"],
        [0, 0],
        [0, 0]
    ),
    # 5. ignore_o=True: Empty result if all are "O"
    (
        True,
        "PER",
        ["O", "O"],
        ["O", "O"],
        [],
        []
    )
]
