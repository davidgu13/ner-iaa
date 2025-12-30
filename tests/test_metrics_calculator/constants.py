from typings.ner_label import NERLabel
from typings.word_span import WordSpan

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
     [
         NERLabel(start_index=0, end_index=4, entity_type="PER"),
         NERLabel(start_index=14, end_index=20, entity_type="LOC")
     ],
     ["PER", "O", "O", "LOC"]
     ),
    # 2. Multi-word entity (Check if it handles sequence of tags)
    ("New York is cold",
     [
         NERLabel(start_index=0, end_index=8, entity_type="LOC")
     ],
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
        [
            NERLabel(start_index=2, end_index=13, entity_type="ORG")
        ],
        ["ORG", "ORG"]
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
    # 3. Keep where only seq2 is non-O
    (
        ["O", "O"],
        ["O", "MISC"],
        ["O"],
        ["MISC"]
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
