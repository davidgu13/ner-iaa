"""
Metrics:
- Mutual F1 score, mutual Cohen's Kappa score
- With O occurrences, without O occurrences
- Accounting for partial overlap
"""
import re
from typing import Callable, Optional

import numpy as np

from metrics import calculate_cohens_kappa, calculate_f1
from typings.entity_type import ENTITY_TYPES
from typings.ner_label import NERLabel
from typings.supported_metrics import SupportedMetrics
from typings.word_span import WordSpan

metric2implementation = {SupportedMetrics.f1: calculate_f1, SupportedMetrics.cohens_kappa: calculate_cohens_kappa}


class MetricsCalculator:
    """
    Assumptions:
    1. Tokenization is space-delimited; Partial overlap doesn't affect the scores
    2. Labels are at word level, e.g. "hu nasa le[pariz]" is forbidden
    3. NER labels are flat, not nested
    TODO:
    - Check the calculations' correctness = add tests
    - Get rid of the space-delimetering (without using BIO) and move to using just the indices of the labels' spans. Is it also the part where partial overlapping is accounted?
    """

    def __init__(self, should_ignore_o_labels: bool = True, partial_overlap_metric: Callable = None):
        self.should_ignore_o_labels = should_ignore_o_labels
        self.partial_overlap_metric = partial_overlap_metric

    @staticmethod
    def _convert_text_to_word_spans(text: str) -> list[WordSpan]:
        """
        For every space-delimited word in the text, return its start & end indices
        :param text: str
        :return: list of the format [(w_1, start_index_1, end_index_1), ..., (wn, start_index_n, end_index_n)]
        """
        tokens_with_indices: list[WordSpan] = []
        # \S+ matches any sequence of non-whitespace characters
        for match in re.finditer(r'\S+', text):
            tokens_with_indices.append(WordSpan.model_validate({
                "text": match.group(),  # The actual text
                "start_index": match.start(),  # Start index (inclusive)
                "end_index": match.end()  # End index (exclusive)
            }))
        return tokens_with_indices

    @staticmethod
    def _convert_labels_to_sequence(text: str, labels: list[NERLabel]) -> list[str]:
        """
        Covnert Doccano's spans format to word-level tags.
        :param text: str
        :param labels: Doccano's spans format, e.g. [[20, 25, PER], [30, 40, LOC], [43, 49, ORG]]
        :return: the tags of each word
        """
        words_spans: list[WordSpan] = MetricsCalculator._convert_text_to_word_spans(text)
        labels_start_indices = [label.start_index for label in labels]

        entity_types_sequence: list[str] = []
        current_label: Optional[NERLabel] = None

        for word_span in words_spans:
            should_be_B_tag = word_span.start_index in labels_start_indices
            should_be_I_tag = word_span in current_label if current_label else False
            if should_be_B_tag or should_be_I_tag:
                try:
                    current_label = [label for label in labels if word_span in label][0]
                except IndexError as e:
                    raise e  # "WTF Bro there's a label-text conflict"
                entity_type = current_label.entity_type
            else:
                entity_type = "O"
            entity_types_sequence.append(entity_type)
        return entity_types_sequence

    @staticmethod
    def _filter_non_o_labels(sequence1: list[str], sequence2: list[str]):
        """
            Returns two lists containing only the elements where
            at least one of the annotators did not label it as "O".
        """
        # Use zip to pair elements, then filter if either isn't "O"
        filtered_pairs = [(a, b) for a, b in zip(sequence1, sequence2) if not (a == "O" and b == "O")]

        # If no entities were found, return two empty lists
        if not filtered_pairs:
            return [], []

        # Unpack the pairs back into two separate lists
        non_o_sequence1, non_o_sequence2 = zip(*filtered_pairs)
        return list(non_o_sequence1), list(non_o_sequence2)

    def _mask_sequence(self, sequence1: list[str], sequence2: list[str], entity: str) -> \
            tuple[list[int], list[int]]:
        """
        Mask two lists of NER labels, with respect to one of the entity types.
        If "O" labels should be ignored, the common "O" labels are filtered out first.
        """
        if self.should_ignore_o_labels:
            non_o_sequence1, non_o_sequence2 = MetricsCalculator._filter_non_o_labels(sequence1, sequence2)
            sequence1_mask = [1 if item == entity else 0 for item in non_o_sequence1]
            sequence2_mask = [1 if item == entity else 0 for item in non_o_sequence2]
        else:
            sequence1_mask = [1 if item == entity else 0 for item in sequence1]
            sequence2_mask = [1 if item == entity else 0 for item in sequence2]
        return sequence1_mask, sequence2_mask

    def _calculate_score(self, entity_type: str, sequence1: list[str], sequence2: list[str], metric: SupportedMetrics):
        # Mask the 2 labels w.r.t. the current entity, so the task is reduced to binary classification
        sequence1_mask, sequence2_mask = self._mask_sequence(sequence1, sequence2, entity_type)
        score_per_entity = metric2implementation[metric](sequence1_mask, sequence2_mask)
        return score_per_entity

    def report_metrics(self, text: str, labels1: list[NERLabel], labels2: list[NERLabel]):
        sequence1 = MetricsCalculator._convert_labels_to_sequence(text, labels1)
        sequence2 = MetricsCalculator._convert_labels_to_sequence(text, labels2)

        scores_per_entity = {}
        for entity_type in ENTITY_TYPES:
            f1_score = self._calculate_score(entity_type, sequence1, sequence2, SupportedMetrics.f1)
            kappa_score = self._calculate_score(entity_type, sequence1, sequence2, SupportedMetrics.cohens_kappa)
            scores_per_entity[entity_type] = {"f1_score": np.round(f1_score, 4),
                                              "cohens_kappa_score": np.round(kappa_score, 4)}
        return scores_per_entity


if __name__ == '__main__':
    input_text = "Paris Whitney Hilton , born February 17, 1981 is an American television " \
                 "personality and businesswoman . She is the great-granddaughter of " \
                 "Conrad Hilton , the founder of Hilton Hotels . Born in New York City and " \
                 "raised in both California and New York , Hilton began a modeling career " \
                 "when she signed with Donald Trump ’s modeling agency ."
    doccano_labels1 = [[0, 20, 'PER'],  # Paris Whitney Hilton
                       [28, 45, 'TEMP'],  # February 17 , 1981
                       [138, 151, 'PER'],  # Conrad Hilton
                       [169, 182, 'ORG'],  # Hilton Hotel
                       [193, 206, 'LOC'],  # New York City
                       [226, 236, 'LOC'],  # California
                       [241, 249, 'LOC'],  # New York
                       [252, 258, 'PER'],  # Hilton
                       # [304, 316, 'PER'],    # Donald Trump
                       [304, 335, 'ORG']]  # Donald Trump ’s modeling agency

    doccano_labels2 = [[0, 20, 'PER'],  # Paris Whitney Hilton
                       [28, 45, 'TEMP'],  # February 17 , 1981
                       [138, 151, 'PER'],  # Conrad Hilton
                       [169, 182, 'ORG'],  # Hilton Hotels
                       [185, 189, 'PER'],  # Born
                       [193, 206, 'LOC'],  # New York City
                       [226, 236, 'LOC'],  # California
                       [241, 249, 'LOC'],  # New York
                       [252, 258, 'PER'],  # Hilton
                       [267, 275, 'LOC']]  # modeling

    parse_doccano_labels = lambda doccano_labels: [NERLabel.model_validate({"start_index": label[0],
                                                                            "end_index": label[1],
                                                                            "entity_type": label[2]
                                                                            }) for label in doccano_labels]
    parsed_doccano_labels1 = parse_doccano_labels(doccano_labels1)
    s = MetricsCalculator._convert_labels_to_sequence(input_text, parsed_doccano_labels1)
    # print(*list(enumerate(zip(input_text.split(" "), s))), sep='\n')

    ann1 = ["O", "O", "PER", "O", "O", "O", "PER", "PER", "O", "O", "ORG", "O"]
    ann2 = ["O", "O", "PER", "PER", "O", "O", "PER", "PER", "O", "O", "O", "O"]

    # print(MetricsCalculator._filter_non_o_labels(ann1, ann2))

    parsed_doccano_labels2 = parse_doccano_labels(doccano_labels2)
    metrics_without_o, metrics_with_o = MetricsCalculator(should_ignore_o_labels=True), MetricsCalculator(
        should_ignore_o_labels=False)

    scores_without_o = metrics_without_o.report_metrics(input_text, parsed_doccano_labels1, parsed_doccano_labels2)
    print(f"Scores without 'O':\n{scores_without_o}")
    scores_with_o = metrics_with_o.report_metrics(input_text, parsed_doccano_labels1, parsed_doccano_labels2)
    print(f"Scores with 'O':\n{scores_with_o}")
