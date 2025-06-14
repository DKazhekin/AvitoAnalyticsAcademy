from random import randint
from typing import Union

from sqlalchemy import Engine, func
from sqlalchemy.orm import Session

from internal.storage.database.sqlalchemy.entity import User as UserDB

from .entity import User, UserNotFound


class AdapterUser:
    def __init__(self, engine: Engine):
        self.engine: Engine = engine

    def get_user_by_id(self, user_id: int) -> User | UserNotFound:
        with Session(self.engine, autocommit=False, autoflush=False) as session:
            db_user = session.query(UserDB).filter(UserDB.User_id == user_id).first()

        if db_user is None:
            return UserNotFound(id=user_id, message=f"User with id {user_id} not found")

        return User(
            id=int(db_user.User_id),
        )

    def get_random_user(self) -> Union[User, UserNotFound]:
        with Session(self.engine, autocommit=False, autoflush=False) as session:
            total_users = session.query(func.count(UserDB.User_id)).scalar()

            if total_users == 0:
                return UserNotFound(id=0, message="No users found in database")

            random_offset = randint(0, total_users - 1)

            db_user = session.query(UserDB).offset(random_offset).first()

        return User(
            id=int(db_user.User_id),
        )
