from pydantic import BaseModel, Field, field_validator


class Span(BaseModel):
    start_index: int = Field(ge=0)
    end_index: int

    @field_validator("end_index")
    @classmethod
    def end_must_be_greater_than_start(cls, end_index, info):
        start_index = info.data.get("start_index")
        if start_index is not None and end_index <= start_index:
            raise ValueError("end_index must be greater than start_index")
        return end_index

    def __contains__(self, item):
        return self.start_index <= item.start_index and item.end_index <= self.end_index
