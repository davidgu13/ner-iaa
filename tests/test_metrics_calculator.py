
import pytest

from metrics_calculator import MetricsCalculator
from typings.word_span import WordSpan


TEXT_TO_WORD_SPANS_CASES = [
    ("Hello world", [
        WordSpan(text="Hello", start_index=0, end_index=5),
        WordSpan(text="world", start_index=6, end_index=11)
    ]), # Standard case
    ("Word1    Word2\tWord3", [
        WordSpan(start_index=0, end_index=5, text="Word1"),
        WordSpan(start_index=9, end_index=14, text="Word2"),
        WordSpan(start_index=15, end_index=20, text="Word3")
    ]), # Standard case
    ("  Trim   me  ", [
        WordSpan(text="Trim", start_index=2, end_index=6),
        WordSpan(text="me", start_index=9, end_index=11)
    ]), # Multiple spaces and padding
    ("Price: $100!", [
        WordSpan(text="Price:", start_index=0, end_index=6),
        WordSpan(text="$100!", start_index=7, end_index=12)
    ]), # Punctuation and special characters
    ("", []), # Edge Case: Empty string
    ("  \n  \t  ", []) # Edge Case: Only whitespace (tabs/newlines)
]


class TestMetricsCalculator:
    @pytest.mark.parametrize("text, expected", TEXT_TO_WORD_SPANS_CASES)
    def test_convert_text_to_word_spans(self, text, expected):
        """Verifies that text is correctly split into WordSpan objects with accurate indices."""
        assert MetricsCalculator._convert_text_to_word_spans(text) == expected
