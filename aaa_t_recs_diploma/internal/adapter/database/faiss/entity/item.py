from dataclasses import dataclass

from numpy import ndarray


@dataclass
class FaissItem:
    id: int
    embedding: ndarray


@dataclass
class FaissItemNotFound:
    id: int
    message: str
