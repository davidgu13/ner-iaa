import pytest
from pydantic import ValidationError

from typings.ner_label import NERLabel


NER_LABEL_NAMES_TEST_CASES = [
    ("per", "PER"),
    ("oRG", "ORG"),
    ("Organization", "ORGANIZATION"),
    ("GPE", "GPE"),
    ("loc_name", "LOC_NAME"),
]

class TestNERLabel:
    def test_ner_label_initialization(self):
        """Tests that NERLabel initializes correctly with valid data."""
        ner = NERLabel(start_index=0, end_index=5, entity_type="person")
        assert ner.start_index == 0
        assert ner.end_index == 5
        assert ner.entity_type == "PERSON"  # Validates normalization happened

    @pytest.mark.parametrize("input_type, expected_output", NER_LABEL_NAMES_TEST_CASES)
    def test_entity_type_normalization(self, input_type, expected_output):
        """Tests that various string formats are correctly normalized to uppercase."""
        ner = NERLabel(start_index=10, end_index=20, entity_type=input_type)
        assert ner.entity_type == expected_output

    def test_entity_type_missing(self):
        """Tests that entity_type is a required field."""
        with pytest.raises(ValidationError) as exc_info:
            # Missing entity_type
            NERLabel(start_index=0, end_index=5)

        # Verify the error is specifically for the missing entity_type
        assert "entity_type" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)

    def test_inheritance_integrity(self):
        """
        Briefly ensures Span logic is still active without exhaustive re-testing.
        This confirms the subclass didn't break base validation.
        """
        with pytest.raises(ValidationError, match="end_index must be greater than or equal start_index"):
            NERLabel(start_index=10, end_index=5, entity_type="ORG")