
import pytest

from metrics_calculator import MetricsCalculator
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


class TestMetricsCalculator:
    @pytest.mark.parametrize("text, expected", TEXT_TO_WORD_SPANS_CASES)
    def test_convert_text_to_word_spans(self, text, expected):
        """Verifies that text is correctly split into WordSpan objects with accurate indices."""
        assert MetricsCalculator._convert_text_to_word_spans(text) == expected

    @pytest.mark.parametrize("text, labels, expected", LABELS_TO_SEQUENCE_CASES)
    def test_convert_labels_to_sequence(self, text, labels, expected):
        """Verifies that character-level spans are correctly mapped to word-level tags."""
        result = MetricsCalculator._convert_labels_to_sequence(text, labels)
        assert result == expected

    def test_convert_labels_to_sequence_raises_error(self):
        """Tests that an IndexError is raised if a word is expected to have a tag but no label matches."""
        # This simulates a case where should_be_B_tag is true but the list comprehension fails
        text = "Microsoft"
        labels = [NERLabel(start_index=0, end_index=5, entity_type="ORG")]

        with pytest.raises(IndexError, match="A label-text conflict found for span"):
            MetricsCalculator._convert_labels_to_sequence(text, labels)
