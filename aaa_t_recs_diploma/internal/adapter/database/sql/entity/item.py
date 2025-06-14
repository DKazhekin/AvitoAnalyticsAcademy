from dataclasses import dataclass


@dataclass
class Item:
    id: int
    title: str
    t_rn: int
    description: str
    d_rn: int


@dataclass
class ItemNotFound:
    id: int
    message: str
