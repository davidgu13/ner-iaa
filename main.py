from ner_inter_annotator_agreement_character_level import NERInterAnnotatorAgreementCharacterLevel
from tests.test_metrics_calculator.constants import REAL_EXAMPLE_DOCCANO_LABELS1, REAL_EXAMPLE_DOCCANO_LABELS2, \
    REAL_EXAMPLE_TEXT
from typings.ner_label import NERLabel


def main():
    metrics_without_o = NERInterAnnotatorAgreementCharacterLevel(should_ignore_o_labels=True)
    scores_without_o = metrics_without_o.report_metrics_from_doccano_labels(REAL_EXAMPLE_TEXT,
                                                                            REAL_EXAMPLE_DOCCANO_LABELS1,
                                                                            REAL_EXAMPLE_DOCCANO_LABELS2)
    print(f"Scores without 'O':")
    for k, v in scores_without_o.items():
        print(f"{k}: {v}")


    metrics_with_o = NERInterAnnotatorAgreementCharacterLevel(should_ignore_o_labels=False)
    scores_with_o = metrics_with_o.report_metrics_from_doccano_labels(REAL_EXAMPLE_TEXT, REAL_EXAMPLE_DOCCANO_LABELS1,
                                                                      REAL_EXAMPLE_DOCCANO_LABELS2)
    print(f"\n\nScores without 'O':")
    for k, v in scores_with_o.items():
        print(f"{k}: {v}")

    # spans1 = NERLabel.from_doccano_format_multiple_labels(
    #     [[10, 15, 'PER'],
    #      [15, 20, 'O'],
    #      [20, 25, 'PER']])
    # spans2 = NERLabel.from_doccano_format_multiple_labels(
    #     [[10, 12, 'PER'],
    #      [12, 21, 'O'],
    #      [21, 25, 'PER']])
    #
    # result_span1 = NERLabel.from_doccano_format_multiple_labels([[10, 15, 'PER'], [20, 25, 'PER']])
    # result_span2 = NERLabel.from_doccano_format_multiple_labels([[10, 12, 'PER'], [12, 15, 'O'], [20, 21, 'O'], [21, 25, 'PER']])
    # print(NERInterAnnotatorAgreementCharacterLevel.filter_mutual_o_spans(spans1, spans2))
    #
    # metrics_without_o = NERInterAnnotatorAgreementCharacterLevel(should_ignore_o_labels=True)
    # print(metrics_without_o.report_metrics_from_doccano_labels(30, spans1))


if __name__ == '__main__':
    main()
