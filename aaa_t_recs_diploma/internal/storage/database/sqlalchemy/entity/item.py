from sqlalchemy import Column, Integer, String

from .base import Base


class Item(Base):
    __tablename__ = "items"

    Item_id = Column(Integer, primary_key=True)
    Title = Column(String)
    t_rn = Column(Integer)
    DescriptionRu = Column(String)
    d_rn = Column(Integer)
