from pydantic import field_validator

from typings.span import Span


class NERLabel(Span):
    entity_type: str

    @field_validator("entity_type", mode="before")
    @classmethod
    def normalize_uppercase(cls, v):
        return v.upper()

    def __eq__(self, other):
        return self.entity_type == other.entity_type and super(NERLabel, self).__eq__(other)

    @classmethod
    def from_doccano_format(cls, doccano_label: list[int, int, str]):
        """
        Initializes NERLabel from [start_index, end_index, entity_type]
        """
        start_index, end_index, entity_type = doccano_label
        return cls(start_index=start_index, end_index=end_index, entity_type=entity_type)

    @classmethod
    def from_doccano_format_multiple_labels(cls, docanno_labels: list[list[int, int, str]]):
        return [NERLabel.from_doccano_format(label) for label in docanno_labels]
