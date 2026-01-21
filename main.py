from metrics_calculator import MetricsCalculator
from metrics_calculator_character_level import MetricsCalculatorCharacterLevel
from tests.test_metrics_calculator.constants import PARSED_DOCCANO_LABELS1, REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT
from typings.ner_label import NERLabel


def main():
    # metrics_without_o = MetricsCalculator(should_ignore_o_labels=True)
    # scores_without_o = metrics_without_o.report_metrics_from_doccano_labels(input_text, doccano_labels1, doccano_labels2)
    # print(f"Scores without 'O':\n{scores_without_o}")
    #
    # metrics_with_o = MetricsCalculator(should_ignore_o_labels=False)
    # scores_with_o = metrics_with_o.report_metrics_from_doccano_labels(input_text, doccano_labels1, doccano_labels2)
    # print(f"Scores with 'O':\n{scores_with_o}")

    # metrics_calculator = MetricsCalculator(character_level_evaluation=True, should_ignore_o_labels=True)

    # filled = MetricsCalculator._fill_with_o_labels(REAL_EXAMPLE_SIMPLE_TOKENIZATION_TEXT,
    #                                                PARSED_DOCCANO_LABELS1)
    # print(*filled, sep='\n')

    spans1 = NERLabel.from_doccano_format_multiple_labels(
        [[10, 15, 'PER'],
         [15, 20, 'O'],
         [20, 25, 'PER']])
    # spans2 = NERLabel.from_doccano_format_multiple_labels(
    #     [[10, 12, 'PER'],
    #      [12, 21, 'O'],
    #      [21, 25, 'PER']])
    #
    # result_span1 = NERLabel.from_doccano_format_multiple_labels([[10, 15, 'PER'], [20, 25, 'PER']])
    # result_span2 = NERLabel.from_doccano_format_multiple_labels([[10, 12, 'PER'], [12, 15, 'O'], [20, 21, 'O'], [21, 25, 'PER']])
    # print(MetricsCalculatorCharacterLevel.filter_mutual_o_spans(spans1, spans2))

    print(dict(enumerate(MetricsCalculatorCharacterLevel._convert_labels_to_sequence(30, spans1))))


if __name__ == '__main__':
    main()
