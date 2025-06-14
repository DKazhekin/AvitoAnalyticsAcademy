from typing import Union

from internal.adapter.database.sql import AdapterUser
from internal.adapter.database.sql.entity import User, UserNotFound


class ServiceUser:
    def __init__(self, adapter_user: AdapterUser):
        self.adapter_user = adapter_user

    def get_user_by_id(self, user_id: int) -> User | UserNotFound:
        return self.adapter_user.get_user_by_id(user_id=user_id)

    def get_random_user(self) -> Union[User, UserNotFound]:
        return self.adapter_user.get_random_user()
