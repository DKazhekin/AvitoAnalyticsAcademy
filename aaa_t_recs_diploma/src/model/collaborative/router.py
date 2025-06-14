import json
from ast import literal_eval
from typing import Dict, List, Optional, Tuple

import pandas as pd


class CollaborativeDataRouter:
    def __init__(
        self,
        user_mapping: Optional[Dict[int, Dict[str, List[int]]]] = None,
        mapper: Optional[Dict[str, List[int]]] = None,
    ):
        self.user_mapping = user_mapping
        self.mapper = mapper

    def predict(self, user_id: int) -> Tuple[int, List[int]]:

        if self.user_mapping is None:
            raise RuntimeError("Dataframe is not loaded")

        last_item_interaction = self.user_mapping[user_id]["last_items"][-1]
        model_microcat_preds = self.user_mapping[user_id]["ranker_scores"][:5]

        return last_item_interaction, model_microcat_preds

    def get_microcat_item_ids(self, microcat_id: int) -> List[int]:

        if self.mapper is None:
            raise RuntimeError("Mapper is not loaded")

        return self.mapper[str(microcat_id)]

    def get_user_hisotry(self, user_id: int) -> List[int]:

        if self.user_mapping is None:
            raise RuntimeError("Dataframe is not loaded")

        last_items = self.user_mapping[user_id]["last_items"]

        return last_items

    @staticmethod
    def load(data_path: str, mapper_path: str):
        df = pd.read_csv(data_path, delimiter=",")

        for col_name in ["last_items", "last_interactions", "preds", "ranker_scores"]:
            df[col_name] = df[col_name].apply(literal_eval)

        user_mapping = {
            row["user_id"]: {
                "last_items": row["last_items"],
                "ranker_scores": row["preds"],
            }
            for _, row in df.iterrows()
        }

        with open(mapper_path, "r") as file:
            mapper = json.load(file)

        return CollaborativeDataRouter(user_mapping=user_mapping, mapper=mapper)
