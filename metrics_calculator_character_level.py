from metrics_calculator import MetricsCalculator
from typings.ner_label import NERLabel
from typings.span import Span


class MetricsCalculatorCharacterLevel(MetricsCalculator):
    @staticmethod
    def _fill_with_o_labels(text_length: int, labels: list[NERLabel]) -> list[NERLabel]:
        """
        Given the text and the labels, turns every span that isn't of an entity into an 'O' label.
        Assumption: labels do not overlap (e.g. NestedNER).
        """
        sorted_labels = sorted(labels, key=lambda x: x.start_index)

        complete_labels = []
        current_pos = 0

        for label in sorted_labels:
            if label.start_index > current_pos:
                complete_labels.append(NERLabel.from_doccano_format([current_pos, label.start_index, "O"]))

            complete_labels.append(label)
            current_pos = label.end_index

        # 4. Check if there is a remaining "O" span after the last entity
        if current_pos < text_length:
            complete_labels.append(NERLabel.from_doccano_format([current_pos, text_length, "O"]))

        return complete_labels

    @staticmethod
    def merge_adjacent_labels(labels: list[NERLabel]) -> list[NERLabel]:
        if not labels:
            return []

        merged = []
        current = labels[0].model_copy()

        for next_label in labels[1:]:
            if next_label.entity_type == current.entity_type and next_label.start_index == current.end_index:
                current.end_index = next_label.end_index
            else:
                merged.append(current)
                current = next_label.model_copy()
        merged.append(current)
        return merged

    @staticmethod
    def filter_mutual_o_spans(list_a: list[NERLabel], list_b: list[NERLabel]) -> tuple[list[NERLabel], list[NERLabel]]:
        # 1. Collect all unique split points (start and end indices) from both lists
        endpoints = set()
        for label in list_a + list_b:
            endpoints.add(label.start_index)
            endpoints.add(label.end_index)

        sorted_points = sorted(list(endpoints))

        # Helper to find which label covers a specific interval [start, end]
        def get_label_for_interval(span_list, start, end):
            for label in span_list:
                if Span(start_index=start, end_index=end) in label:
                    return label.entity_type
            return None

        result_a, result_b = [], []

        # 2. Iterate through every atomic interval created by the endpoints
        for i in range(len(sorted_points) - 1):
            start, end = sorted_points[i], sorted_points[i+1]

            type_a = get_label_for_interval(list_a, start, end)
            type_b = get_label_for_interval(list_b, start, end)

            # 3. Filter: Only keep if at least one of them is NOT "O"
            if type_a == "O" and type_b == "O":
                continue

            if type_a is not None:
                result_a.append(NERLabel(start_index=start, end_index=end, entity_type=type_a))
            if type_b is not None:
                result_b.append(NERLabel(start_index=start, end_index=end, entity_type=type_b))

        # 4. Optional: Merge adjacent identical labels to clean up the output
        return MetricsCalculatorCharacterLevel.merge_adjacent_labels(result_a), \
               MetricsCalculatorCharacterLevel.merge_adjacent_labels(result_b)

    @staticmethod
    def _convert_labels_to_sequence(text_length: int, labels: list[NERLabel]) -> list[str]:
        """
        Create a list of the labels in each index of the text.
        """
        entity_type_per_character = []
        labels_with_o = MetricsCalculatorCharacterLevel._fill_with_o_labels(text_length, labels)
        for label in labels_with_o:
            entity_type_per_character.extend([label.entity_type] * (label.end_index - label.start_index))
        return entity_type_per_character

