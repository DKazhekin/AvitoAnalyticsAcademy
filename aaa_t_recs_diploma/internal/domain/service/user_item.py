from typing import List, Tuple, Union

from internal.adapter.database.sql.entity import Item, ItemNotFound, User, UserNotFound
from src.model.collaborative import CollaborativeDataRouter

from .item import ServiceItem
from .user import ServiceUser


class ServiceUserItem:
    def __init__(
        self,
        service_user: ServiceUser,
        service_item: ServiceItem,
        cb_data_router: CollaborativeDataRouter,
    ):
        self.service_user = service_user
        self.service_item = service_item
        self.cb_data_router = cb_data_router

    def recommendation_item(self, user_id: int, N: int = 2) -> Tuple[
        Union[List[Union[Item, ItemNotFound]], UserNotFound],
        List[Union[Item, ItemNotFound]],
    ]:
        user: User | UserNotFound = self.service_user.get_user_by_id(user_id)
        if isinstance(user, UserNotFound):
            return user, []

        last_contact, recommendation_microcat_ids = self.cb_data_router.predict(
            user_id=user_id
        )

        item_ids: List[int] = []
        for microcat_id in recommendation_microcat_ids:
            microcat_item_ids = self.cb_data_router.get_microcat_item_ids(microcat_id)
            rec_item_ids: List[int] = (
                self.service_item.recommend_item_ids_by_embeddings(
                    item_ids=[last_contact], k=N, select_ids=microcat_item_ids
                )[last_contact]
            )

            item_ids.extend(rec_item_ids)

        rec_count = len(item_ids)

        history_item_ids = self.get_user_history_ids(user_id=user_id)

        items = self.service_item.get_items_by_ids(item_ids=item_ids + history_item_ids)

        return items[:rec_count], items[rec_count:]

    def get_user_history_ids(self, user_id: int) -> List[int]:
        return self.cb_data_router.get_user_hisotry(user_id)

    def get_user_history(self, user_id: int) -> List[Union[Item, ItemNotFound]]:
        user_hisotry = self.get_user_history_ids(user_id=user_id)

        items = self.service_item.get_items_by_ids(user_hisotry)

        return items

    def get_random_user_with_history(
        self,
    ) -> Tuple[Union[User, UserNotFound], List[Union[Item, ItemNotFound]]]:
        random_user = self.service_user.get_random_user()

        if isinstance(random_user, UserNotFound):
            return random_user, []

        history_items = self.get_user_history(user_id=random_user.id)

        return random_user, history_items
