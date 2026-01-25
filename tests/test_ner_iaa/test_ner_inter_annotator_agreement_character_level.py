import pytest

from ner_inter_annotator_agreement_character_level import NERInterAnnotatorAgreementCharacterLevel
from typings.ner_label import NERLabel

REAL_EXAMPLE_TEXT = "Paris Whitney Hilton, born February 17, 1981 is an American television " \
                    "personality and businesswoman. She is the great-granddaughter of " \
                    "Conrad Hilton, the founder of Hilton Hotels. Born in New York City and " \
                    "raised in both California and New York, Hilton began a modeling career " \
                    "when she signed with Donald Trump’s modeling agency."

REAL_EXAMPLE_DOCCANO_LABELS1 = [[0, 19, 'PER'],  # Paris Whitney Hilton
                                [27, 43, 'TEMP'],  # February 17 , 1981
                                [136, 148, 'PER'],  # Conrad Hilton
                                [166, 178, 'ORG'],  # Hilton Hotels
                                [189, 201, 'LOC'],  # New York City
                                [222, 231, 'LOC'],  # California
                                [237, 244, 'LOC'],  # New York
                                [247, 252, 'PER'],  # Hilton
                                [299, 328, 'ORG']]  # Donald Trump’s modeling agency

EXPECTED_LABELS_PER_TEXT_INDEX = ['PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER',
                                  'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP',
                                  'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'TEMP', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'PER',
                                  'PER', 'PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG',
                                  'ORG', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'LOC', 'LOC',
                                  'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'O', 'O',
                                  'O', 'O', 'O', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'O', 'O',
                                  'PER', 'PER', 'PER', 'PER', 'PER', 'PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                                  'O', 'O', 'O', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG',
                                  'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG',
                                  'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'ORG', 'O']


@pytest.fixture
def ner_iaa_char_level():
    return NERInterAnnotatorAgreementCharacterLevel()


class TestNERInterAnnotatorAgreementCharLevel:
    def test_convert_labels_to_sequence(self, ner_iaa_char_level):
        """Verifies that character-level spans are correctly mapped to word-level tags."""
        actual_labels_per_text_index = ner_iaa_char_level._convert_labels_to_sequence(
            REAL_EXAMPLE_TEXT,
            NERLabel.from_doccano_format_multiple_labels(REAL_EXAMPLE_DOCCANO_LABELS1))
        assert actual_labels_per_text_index == EXPECTED_LABELS_PER_TEXT_INDEX
