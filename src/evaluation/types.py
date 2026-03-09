from dataclasses import dataclass


@dataclass
class IndexEntry:
    context: tuple[int, ...] | list[int]
    target: tuple[int, ...] | list[int]
