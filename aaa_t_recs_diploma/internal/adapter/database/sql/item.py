from random import randint
from typing import List, Union

from sqlalchemy import Engine, func
from sqlalchemy.orm import Session

from internal.storage.database.sqlalchemy.entity import Item as ItemDB

from .entity import Item, ItemNotFound


class AdapterItem:
    def __init__(self, engine: Engine):
        self.engine: Engine = engine

    def get_item_by_id(self, item_id: int) -> Union[Item, ItemNotFound]:
        with Session(self.engine, autocommit=False, autoflush=False) as session:
            db_item = session.query(ItemDB).filter(ItemDB.Item_id == item_id).first()

        if db_item is None:
            return ItemNotFound(id=item_id, message=f"Item with id {item_id} not found")

        return Item(
            id=int(db_item.Item_id),
            title=str(db_item.Title),
            t_rn=int(db_item.t_rn),
            description=str(db_item.DescriptionRu),
            d_rn=int(db_item.d_rn),
        )

    def get_items_by_ids(self, item_ids: List[int]) -> List[Union[Item, ItemNotFound]]:
        with Session(self.engine, autocommit=False, autoflush=False) as session:
            db_items = session.query(ItemDB).filter(ItemDB.Item_id.in_(item_ids)).all()

        items_dict = {item.Item_id: item for item in db_items}

        results: List[Union[Item, ItemNotFound]] = []
        for item_id in item_ids:
            if item_id in items_dict:
                db_item = items_dict[item_id]
                results.append(
                    Item(
                        id=int(db_item.Item_id),
                        title=str(db_item.Title),
                        t_rn=int(db_item.t_rn),
                        description=str(db_item.DescriptionRu),
                        d_rn=int(db_item.d_rn),
                    )
                )
            else:
                results.append(
                    ItemNotFound(
                        id=item_id, message=f"Item with id {item_id} not found"
                    )
                )

        return results

    def get_random_item(self) -> Union[Item, ItemNotFound]:
        with Session(self.engine, autocommit=False, autoflush=False) as session:
            total_items = session.query(func.count(ItemDB.Item_id)).scalar()

            if total_items == 0:
                return ItemNotFound(id=0, message="No items found in database")

            random_offset = randint(0, total_items - 1)

            db_item = session.query(ItemDB).offset(random_offset).first()

        return Item(
            id=int(db_item.Item_id),
            title=str(db_item.Title),
            t_rn=int(db_item.t_rn),
            description=str(db_item.DescriptionRu),
            d_rn=int(db_item.d_rn),
        )
