from enum import Enum


class EntityType(str, Enum):
    PER = "PER"
    LOC = "LOC"
    ORG = "ORG"
    TEMP = "TEMP"


ENTITY_TYPES = [member.value for member in EntityType]
