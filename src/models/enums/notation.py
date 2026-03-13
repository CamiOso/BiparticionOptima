from enum import Enum


class Notation(Enum):
    """Notaciones para indexacion de estados."""

    LIL_ENDIAN = "little-endian"
    BIG_ENDIAN = "big-endian"
    GRAY_CODE = "gray-code"
