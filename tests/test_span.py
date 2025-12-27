import pytest
from pydantic import ValidationError

from typings.span import Span

SPANS_CONTAINMENT_LOGIC_CASES = [
    (Span(start_index=0, end_index=10), Span(start_index=2, end_index=5), True),  # Fully inside
    (Span(start_index=0, end_index=10), Span(start_index=0, end_index=10), True),  # Identical
    (Span(start_index=5, end_index=10), Span(start_index=4, end_index=8), False),  # Overlap left
    (Span(start_index=5, end_index=10), Span(start_index=7, end_index=12), False),  # Overlap right
    (Span(start_index=5, end_index=10), Span(start_index=0, end_index=4), False),  # Completely outside
]


class TestSpan:
    def test_valid_span(self):
        """Tests that a valid span is created correctly."""
        span = Span(start_index=0, end_index=10)
        assert span.start_index == 0
        assert span.end_index == 10

    def test_negative_start_index(self):
        """Tests the pydantic Field(ge=0) constraint."""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            # the error message can be accessed via exc.errors()[0]["msg"]
            Span(start_index=-1, end_index=10)

    def test_end_index_less_than_start(self):
        """Tests the custom field_validator logic."""
        with pytest.raises(ValidationError, match="end_index must be greater than start_index"):
            Span(start_index=5, end_index=3)

    def test_end_index_equal_to_start(self):
        """Tests that end_index cannot be equal to start_index."""
        with pytest.raises(ValidationError, match="end_index must be greater than start_index"):
            Span(start_index=5, end_index=5)

    @pytest.mark.parametrize("parent, child, expected", SPANS_CONTAINMENT_LOGIC_CASES)
    def test_contains_logic(self, parent, child, expected):
        """Tests the __contains__ method with various overlap scenarios."""
        assert (child in parent) is expected
