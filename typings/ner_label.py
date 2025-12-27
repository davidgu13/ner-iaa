from pydantic import field_validator

from typings.span import Span


class NERLabel(Span):
    entity_type: str

    @field_validator("entity_type", mode="before")
    @classmethod
    def normalize_uppercase(cls, v):
        return v.upper()
