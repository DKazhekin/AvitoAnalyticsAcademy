from typing import Dict, List, Tuple, Union

from internal.adapter.database.sql.entity import Item, ItemNotFound, User, UserNotFound
from internal.domain.service import ServiceUserItem


class HandlerUserItem:
    def __init__(self, service_user_item: ServiceUserItem):
        self.service_user_item = service_user_item

    def recommendation_item(self, user_id: int) -> Union[
        Tuple[List[Dict[str, str | int]], List[Dict[str, str | int]]],
        Dict[str, str | int],
    ]:
        service_response = self.service_user_item.recommendation_item(user_id)
        if isinstance(service_response[0], UserNotFound):
            user: UserNotFound = service_response[0]
            return {"id": user.id, "message": user.message}

        rec_items: List[Item | ItemNotFound] = service_response[0]
        history_items: List[Item | ItemNotFound] = service_response[1]

        return [
            (
                {
                    "id": rec_item.id,
                    "title": rec_item.title,
                    "description": rec_item.description,
                }
                if isinstance(rec_item, Item)
                else {"id": rec_item.id, "message": rec_item.message}
            )
            for rec_item in rec_items
        ], [
            (
                {
                    "id": history_item.id,
                    "title": history_item.title,
                    "description": history_item.description,
                }
                if isinstance(history_item, Item)
                else {"id": history_item.id, "message": history_item.message}
            )
            for history_item in history_items
        ]

    def get_random_user_with_history(
        self,
    ) -> Union[
        Tuple[Dict[str, Union[str, int]], List[Dict[str, Union[str, int]]]],
        Dict[str, Union[str, int]],
    ]:
        service_response = self.service_user_item.get_random_user_with_history()

        if isinstance(service_response[0], UserNotFound):
            not_found_user: UserNotFound = service_response[0]
            return {"id": not_found_user.id, "message": not_found_user.message}

        user: User = service_response[0]
        history_items: List[Union[Item, ItemNotFound]] = service_response[1]

        return {"id": user.id}, [
            (
                {
                    "id": history_item.id,
                    "title": history_item.title,
                    "description": history_item.description,
                }
                if isinstance(history_item, Item)
                else {"id": history_item.id, "message": history_item.message}
            )
            for history_item in history_items
        ]
