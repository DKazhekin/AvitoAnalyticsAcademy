from typing import Dict, Union

from internal.adapter.database.sql.entity import User, UserNotFound
from internal.domain.service import ServiceUser


class HandlerUser:
    def __init__(self, service_user: ServiceUser):
        self.service_user = service_user

    def get_user_by_id(self, user_id: int) -> Dict[str, Union[str, int]]:
        user: User | UserNotFound = self.service_user.get_user_by_id(user_id=user_id)
        if isinstance(user, UserNotFound):
            return {"id": user.id, "message": user.message}

        return {
            "id": user.id,
        }

    def get_random_user(self) -> Dict[str, Union[str, int]]:
        user: User | UserNotFound = self.service_user.get_random_user()
        if isinstance(user, UserNotFound):
            return {"id": user.id, "message": user.message}

        return {
            "id": user.id,
        }
