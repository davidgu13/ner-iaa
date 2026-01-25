from ner_inter_annotator_agreement_character_level import NERInterAnnotatorAgreementCharacterLevel
from tests.test_ner_iaa.constants import REAL_EXAMPLE_DOCCANO_LABELS1, REAL_EXAMPLE_DOCCANO_LABELS2, \
    REAL_EXAMPLE_TEXT


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


if __name__ == '__main__':
    main()
