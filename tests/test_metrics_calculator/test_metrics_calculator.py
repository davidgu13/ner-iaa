import pytest

from metrics_calculator import MetricsCalculator
from tests.test_metrics_calculator.constants import FILTER_NON_O_LABELS_CASES, LABELS_TO_SEQUENCE_CASES, \
    MASK_SEQUENCE_CASES, PARSED_DOCCANO_LABELS1, PARSED_DOCCANO_LABELS2, \
    REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS1, \
    REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS2, REAL_EXAMPLE_SIMPLE_TOKENIZATION_EXPECTED_SCORES_WITHOUT_O, \
    REAL_EXAMPLE_SIMPLE_TOKENIZATION_EXPECTED_SCORES_WITH_O, REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT, \
    TEXT_TO_WORD_SPANS_CASES
from typings.ner_label import NERLabel


class MockCalculator:
    """A small mock to simulate the instance containing the flag."""

    def __init__(self, should_ignore_o_labels: bool):
        self.should_ignore_o_labels = should_ignore_o_labels

    # Binding the method to the mock for testing
    _mask_sequence = MetricsCalculator._mask_sequence


@pytest.fixture
def metrics_calculator():
    return NERInterAnnotatorAgreement()


class TestMetricsCalculator:
    @pytest.mark.parametrize("text, expected", TEXT_TO_WORD_SPANS_CASES)
    def test_convert_text_to_word_spans(self, metrics_calculator, text, expected):
        """Verifies that text is correctly split into WordSpan objects with accurate indices."""
        assert metrics_calculator._convert_text_to_word_spans(text) == expected

    @pytest.mark.parametrize("text, labels, expected", LABELS_TO_SEQUENCE_CASES)
    def test_convert_labels_to_sequence(self, metrics_calculator, text, labels, expected):
        """Verifies that character-level spans are correctly mapped to word-level tags."""
        result = metrics_calculator._convert_labels_to_sequence(text, labels)
        assert result == expected

    def test_convert_labels_to_sequence_raises_error(self, metrics_calculator):
        """Tests that an IndexError is raised if a word is expected to have a tag but no label matches."""
        # This simulates a case where should_be_B_tag is true but the list comprehension fails
        text = "Microsoft"
        labels = [NERLabel(start_index=0, end_index=5, entity_type="ORG")]

        with pytest.raises(IndexError, match="A label-text conflict found for span"):
            metrics_calculator._convert_labels_to_sequence(text, labels)

    @pytest.mark.parametrize("seq1, seq2, expected1, expected2", FILTER_NON_O_LABELS_CASES)
    def test_filter_non_o_labels(self, seq1, seq2, expected1, expected2):
        """Verifies that pairs are only kept if at least one element is not 'O'."""
        res1, res2 = MetricsCalculator._filter_non_o_labels(seq1, seq2)
        assert res1 == expected1
        assert res2 == expected2

    def test_filter_non_o_labels_length_mismatch(self):
        """Verifies that a ValueError is raised when sequences have different lengths."""
        seq1 = ["O", "PER", "LOC"]
        seq2 = ["O", "PER"]

        with pytest.raises(ValueError, match="Sequences have different lengths"):
            MetricsCalculator._filter_non_o_labels(seq1, seq2)

    @pytest.mark.parametrize("ignore_o, entity, seq1, seq2, expected1, expected2", MASK_SEQUENCE_CASES)
    def test_mask_sequence(self, ignore_o, entity, seq1, seq2, expected1, expected2):
        calc = MockCalculator(should_ignore_o_labels=ignore_o)
        res1, res2 = calc._mask_sequence(seq1, seq2, entity)

        assert res1 == expected1
        assert res2 == expected2

    def test_report_metrics_real_example_without_o(self):
        metrics_without_o = MetricsCalculator(should_ignore_o_labels=True)
        actual_scores_without_o = metrics_without_o._report_metrics_from_labels(REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT,
                                                                                PARSED_DOCCANO_LABELS1,
                                                                                PARSED_DOCCANO_LABELS2)
        assert REAL_EXAMPLE_SIMPLE_TOKENIZATION_EXPECTED_SCORES_WITHOUT_O == actual_scores_without_o

    def test_report_metrics_real_example_with_o(self):
        metrics_with_o = MetricsCalculator(should_ignore_o_labels=False)
        actual_scores_with_o = metrics_with_o._report_metrics_from_labels(REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT,
                                                                          PARSED_DOCCANO_LABELS1,
                                                                          PARSED_DOCCANO_LABELS2)
        assert REAL_EXAMPLE_SIMPLE_TOKENIZATION_EXPECTED_SCORES_WITH_O == actual_scores_with_o

    def test_report_metrics_real_example_without_o_from_doccano(self):
        metrics_without_o = MetricsCalculator(should_ignore_o_labels=True)
        actual_scores_without_o = metrics_without_o.report_metrics_from_doccano_labels(
            REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT,
            REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS1,
            REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS2)
        assert REAL_EXAMPLE_SIMPLE_TOKENIZATION_EXPECTED_SCORES_WITHOUT_O == actual_scores_without_o

    def test_report_metrics_real_example_with_o_from_doccano(self):
        metrics_with_o = MetricsCalculator(should_ignore_o_labels=False)
        actual_scores_with_o = metrics_with_o.report_metrics_from_doccano_labels(
            REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT,
            REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS1,
            REAL_EXAMPLE_SIMPLE_TOKENIZATION_DOCCANO_LABELS2)
        assert REAL_EXAMPLE_SIMPLE_TOKENIZATION_EXPECTED_SCORES_WITH_O == actual_scores_with_o
