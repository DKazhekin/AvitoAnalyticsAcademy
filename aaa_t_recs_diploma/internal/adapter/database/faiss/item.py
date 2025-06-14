from typing import Dict, List, Union

import faiss
import numpy as np
from faiss import Index

from .entity import FaissItem, FaissItemNotFound


class AdapterFaissItem:
    def __init__(self, faiss_index: Index):
        self.faiss_index = faiss_index

    def get_similar_item_ids(
        self,
        items: List[Union[FaissItem, FaissItemNotFound]],
        k: int,
        select_ids: List[int] = [],
    ) -> Dict[int, List[int]]:
        not_found_items = [
            item for item in items if isinstance(item, FaissItemNotFound)
        ]
        if len(not_found_items) == len(items):
            return {not_found_item.id: [] for not_found_item in not_found_items}

        found_items: List[FaissItem] = [
            item for item in items if isinstance(item, FaissItem)
        ]
        embeddings = np.stack([item.embedding for item in found_items], axis=0)
        params = None
        if len(select_ids) != 0:
            sel = faiss.IDSelectorBatch(select_ids)
            params = faiss.SearchParameters(sel=sel)

        _, rec_ids = self.faiss_index.search(embeddings, k, params=params)

        ids = [item.id for item in found_items]
        similar_item_ids = {idx: rec_ids[i].tolist() for i, idx in enumerate(ids)}
        similar_item_ids.update(
            {not_found_item.id: [] for not_found_item in not_found_items}
        )

        return similar_item_ids

    def get_items_by_ids(
        self, item_ids: List[int]
    ) -> List[Union[FaissItem, FaissItemNotFound]]:
        try:
            embeddings = self.faiss_index.reconstruct_batch(item_ids)
            return [
                FaissItem(id=idx, embedding=emb)
                for idx, emb in zip(item_ids, embeddings)
            ]
        except (RuntimeError, AttributeError):
            items: List[Union[FaissItem, FaissItemNotFound]] = []
            for idx in item_ids:
                try:
                    items.append(
                        FaissItem(id=idx, embedding=self.faiss_index.reconstruct(idx))
                    )
                except RuntimeError:
                    items.append(
                        FaissItemNotFound(id=idx, message="faiss item not found")
                    )
        return items

    def add_item_embedding(self, item: FaissItem) -> None:
        self.faiss_index.add_with_ids(
            item.embedding.unsqueeze(0).numpy(), np.array([item.id])
        )

        return None

    def add_items_embedding(self, items: List[FaissItem]) -> None:
        ids = np.array([item.id for item in items])
        embeddings = np.stack([item.embedding for item in items], axis=0)
        self.faiss_index.add_with_ids(embeddings, ids)

        return None
