from sqlalchemy import Column, Integer

from .base import Base


class User(Base):
    __tablename__ = "users"

    User_id = Column(Integer, primary_key=True)
