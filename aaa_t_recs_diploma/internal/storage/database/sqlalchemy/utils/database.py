import pandas as pd
from sqlalchemy import Engine

from internal.storage.database.sqlalchemy.entity import Base


def db_from_df(engine: Engine, item_df: pd.DataFrame, user_df: pd.DataFrame):
    Base.metadata.create_all(bind=engine)
    item_df.to_sql("items", con=engine, if_exists="replace", index=False)
    user_df.to_sql("users", con=engine, if_exists="replace", index=False)


def prepare_df(item_df_path: str, user_df_path: str):
    item_df = pd.read_parquet(item_df_path).rename(columns={"item_id": "Item_id"})
    user_df = pd.read_csv(user_df_path)
    user_df = user_df[["user_id"]].drop_duplicates(keep=False)
    user_df.columns = ["User_id"]

    return item_df, user_df
