from typing import Dict, List, Union

from internal.adapter.database.sql.entity import Item, ItemNotFound
from internal.domain.service import ServiceItem


class HandlerItem:
    def __init__(self, service_item: ServiceItem):
        self.service_item = service_item

    def get_item_by_id(self, item_id: int) -> Dict[str, Union[str, int]]:
        item: Item | ItemNotFound = self.service_item.get_item_by_id(item_id)
        if isinstance(item, ItemNotFound):
            return {"id": item.id, "message": item.message}

        return {
            "id": item.id,
            "title": item.title,
            "description": item.description,
        }

    def recommend_items_by_embedding(
        self, item_id: int
    ) -> Dict[str, Union[str, int, List[Dict[str, Union[str, int]]]]]:
        recommend = self.service_item.recommend_items_by_embeddings(item_ids=[item_id])
        item_id, rec_items = list(recommend.items())[0]
        rec_items_dict: List[Dict[str, Union[str, int]]] = [
            (
                {
                    "id": rec_item.id,
                    "title": rec_item.title,
                    "description": rec_item.description,
                }
                if isinstance(rec_item, Item)
                else {
                    "id": rec_item.id,
                    "message": rec_item.message,
                }
            )
            for rec_item in rec_items
        ]

        return {"id": item_id, "rec_items": rec_items_dict}

    def get_random_item(self) -> Dict[str, Union[str, int]]:
        item: Item | ItemNotFound = self.service_item.get_random_item()
        if isinstance(item, ItemNotFound):
            return {"id": item.id, "message": item.message}

        return {
            "id": item.id,
            "title": item.title,
            "description": item.description,
        }
