from dataclasses import dataclass


@dataclass
class User:
    id: int


@dataclass
class UserNotFound:
    id: int
    message: str
