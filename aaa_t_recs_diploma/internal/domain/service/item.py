from typing import Dict, List, Union

from internal.adapter.database.faiss import AdapterFaissItem
from internal.adapter.database.sql import AdapterItem
from internal.adapter.database.sql.entity import Item, ItemNotFound


class ServiceItem:
    def __init__(self, adapter_item: AdapterItem, adapter_faiss_item: AdapterFaissItem):
        self.adapter_item = adapter_item
        self.adapter_faiss_item = adapter_faiss_item

    def get_item_by_id(self, item_id: int) -> Union[Item, ItemNotFound]:
        return self.adapter_item.get_item_by_id(item_id=item_id)

    def get_items_by_ids(self, item_ids: List[int]) -> List[Union[Item, ItemNotFound]]:
        return self.adapter_item.get_items_by_ids(item_ids=item_ids)

    def recommend_item_ids_by_embeddings(
        self, item_ids: List[int], k: int = 5, select_ids: List[int] = []
    ) -> Dict[int, List[int]]:
        faiss_items = self.adapter_faiss_item.get_items_by_ids(item_ids=item_ids)
        rec_item_ids = self.adapter_faiss_item.get_similar_item_ids(
            items=faiss_items, k=k, select_ids=select_ids
        )

        return rec_item_ids

    def recommend_items_by_embeddings(
        self, item_ids: List[int], k: int = 5, select_ids: List[int] = []
    ) -> Dict[int, List[Union[Item, ItemNotFound]]]:
        rec_item_ids = self.recommend_item_ids_by_embeddings(
            item_ids=item_ids, k=k, select_ids=select_ids
        )

        return {
            item_id: self.get_items_by_ids(item_ids=rec_item_id)
            for item_id, rec_item_id in rec_item_ids.items()
        }

    def get_random_item(self) -> Union[Item, ItemNotFound]:
        return self.adapter_item.get_random_item()
