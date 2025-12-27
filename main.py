from metrics_calculator import MetricsCalculator
from typings.ner_label import NERLabel


def main():
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


if __name__ == '__main__':
    main()
