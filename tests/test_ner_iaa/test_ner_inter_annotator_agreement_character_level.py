import pytest

from ner_inter_annotator_agreement_character_level import NERInterAnnotatorAgreementCharacterLevel
from tests.test_ner_iaa.constants import LABELS_TO_SEQUENCE_CASES


@pytest.fixture
def ner_iaa_char_level():
    return NERInterAnnotatorAgreementCharacterLevel()


class TestNERInterAnnotatorAgreementCharLevel:
    def test_convert_labels_to_sequence(self, ner_iaa_char_level):
        """Verifies that character-level spans are correctly mapped to word-level tags."""
        result = ner_iaa_char_level._convert_labels_to_sequence(text, labels)
        assert result == expected
