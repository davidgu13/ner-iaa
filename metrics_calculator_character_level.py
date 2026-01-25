from metrics_calculator import NERInterAnnotatorAgreement
from typings.ner_label import NERLabel


class NERInterAnnotatorAgreementCharacterLevel(NERInterAnnotatorAgreement):
    """
    No need to override the methods report_metrics_from_doccano_labels and _report_metrics_from_labels.
    The only difference is in _convert_labels_to_sequence, and it derives from the "character-level tokenization"
    """

    @staticmethod
    def _fill_implicit_o_labels(text_length: int, labels: list[NERLabel]) -> list[NERLabel]:
        """
        Given the text and the labels, turns every span that isn't of an entity into an 'O' label.
        Assumption: labels do not overlap (e.g. NestedNER).
        """
        sorted_labels = sorted(labels, key=lambda x: x.start_index)

        explicit_labels = []
        current_pos = 0

        for label in sorted_labels:
            if label.start_index > current_pos:
                explicit_labels.append(NERLabel.from_doccano_format([current_pos, label.start_index - 1, "O"]))

            explicit_labels.append(label)
            current_pos = label.end_index + 1

        # 4. Check if there is a remaining "O" span after the last entity
        if current_pos < text_length:
            explicit_labels.append(NERLabel.from_doccano_format([current_pos, text_length - 1, "O"]))

        return explicit_labels

    def _convert_labels_to_sequence(self, text: str, labels: list[NERLabel]) -> list[str]:
        """
        Create a character-level list of labels, based on the text.
        For example, "NYC is..." -> [LOC, LOC, LOC, O, O, O, ...]
        """
        character_level_labels = []
        explicit_labels = NERInterAnnotatorAgreementCharacterLevel._fill_implicit_o_labels(len(text), labels)
        for label in explicit_labels:
            character_level_labels.extend([label.entity_type] * len(label))
        return character_level_labels
