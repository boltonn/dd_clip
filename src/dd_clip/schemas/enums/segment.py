from enum import StrEnum


class SegmentStrategy(StrEnum):
    default = "default"
    recursive = "recursive"
    pysbd = "pysbd"
    pysbd_merge = "pysbd_merge"
