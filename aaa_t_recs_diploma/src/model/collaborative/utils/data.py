import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix


# Финализация весов для тренировочной выборки катбуста
def create_als_train_data(data):
    train_data_new = defaultdict(list)

    for _, tdf in data.groupby(["user_id", "microcat_id"]):

        tdf = tdf.sort_values("event_date")
        sample = tdf.iloc[-1, :]

        train_data_new["user_id"].append(sample.user_id)
        train_data_new["item_id"].append(sample.item_id)
        train_data_new["microcat_id"].append(sample.microcat_id)
        train_data_new["event_date"].append(sample.event_date)
        train_data_new["eid"].append(sample.eid)
        train_data_new["user_idx"].append(sample.user_idx)
        train_data_new["microcat_idx"].append(sample.microcat_idx)
        train_data_new["weight"].append(sum(tdf.weight.tolist()))

    train_data = pd.DataFrame(train_data_new)
    return train_data


# Затухание веса со временем
def compute_decay_weight(row, last_interactions, beta=0.3):
    user_id = row.user_id
    cur_interaction = row.event_date

    delta_days = (
        last_interactions[last_interactions["user_id"] == user_id].event_date.iloc[0]
        - cur_interaction
    ).days
    return 1 / (1 + beta * delta_days)  # - гиперболическое, менее сильное затухание


# Собираем последние 5 мирокатегорий для каждого user_id
def get_last_interactions(user_id, last_interactions_frame):
    res = (
        last_interactions_frame[last_interactions_frame["user_id"] == user_id]
        .iloc[0]
        .last_interactions
    )  # last_interactions_valid создавали на этапе сбора данных для получения последних 5 взаимодействий по микрокатегориям для каждого пользователя
    while len(res) != 5:
        res.append(-1)
    return res


# Считает ALS скор между пользователем и конкретной микрокатегорией
def get_als_score(user_idx, microcat_idx, user_vectors, item_vectors):
    score = np.dot(user_vectors[user_idx], item_vectors[microcat_idx])
    return score


# Считает среднее представление последних 5 микрокатегорий
def get_mean_last_microcats_embed(microcat_ids, item_vectors):
    result_embed = item_vectors[microcat_ids].mean(axis=0).tolist()
    return result_embed


# Дает вектор пользователя
def get_user_embed(user_idx, user_vectors):
    return user_vectors[user_idx].tolist()


# Дает вектор микрокатегории
def get_microcat_embed(item_idx, item_vectors):
    return item_vectors[item_idx].tolist()


# Функция для сбора датасета под трейн катбуста
def create_catboost_train_data(
    valid_frame,
    user_to_index,
    microcat_to_index,
    user_vectors,
    item_vectors,
    last_interactions_valid,
    all_train_microcats,
):

    info = defaultdict(list)
    user_positive_microcats = defaultdict(list)

    for i, row in valid_frame.iterrows():

        info["user_id"].append(str(row.user_idx))

        # последние взаимодействия по микрокатегориям
        last_interactions = get_last_interactions(row.user_id, last_interactions_valid)
        for i, elem in enumerate(
            list(map(lambda x: str(microcat_to_index[x]), last_interactions))
        ):
            info[f"last_microcat_{i + 1}"].append(elem)

        # усредненный вектор последних 5 микрокатегорий
        mean_microcat_embed = get_mean_last_microcats_embed(
            [microcat_to_index[elem] for elem in last_interactions if elem != -1],
            item_vectors,
        )
        for i, elem in enumerate(mean_microcat_embed):
            info[f"last_microcat_embed_value_{i + 1}"].append(elem)

        # вектор пользователя
        user_embed = get_user_embed(row.user_idx, user_vectors)
        for i, elem in enumerate(user_embed):
            info[f"user_embed_value_{i + 1}"].append(elem)

        # вектор таргет микрокатегории
        item_embed = get_microcat_embed(row.microcat_idx, item_vectors)
        for i, elem in enumerate(item_embed):
            info[f"item_embed_value_{i + 1}"].append(elem)

        info["interaction_type"].append(str(row.eid))
        info["target_microcat"].append(str(row.microcat_idx))
        info["target_score"].append(
            get_als_score(row.user_idx, row.microcat_idx, user_vectors, item_vectors)
        )
        info["target"].append(1)

        user_positive_microcats[row.user_id].append(row.microcat_idx)

    all_eids = valid_frame.eid.unique().tolist()
    for user_id in user_positive_microcats:

        positive_microcats = set(user_positive_microcats[user_id])
        for negative_microcat in random.sample(
            sorted(all_train_microcats - positive_microcats), 10
        ):
            new_negative_microcat = microcat_to_index[negative_microcat]

            info["user_id"].append(str(user_to_index[user_id]))

            # последние взаимодействия по микрокатегориям
            last_interactions = get_last_interactions(user_id, last_interactions_valid)
            for i, elem in enumerate(
                list(map(lambda x: str(microcat_to_index[x]), last_interactions))
            ):
                info[f"last_microcat_{i + 1}"].append(elem)

            # усредненный вектор последних 5 микрокатегорий
            mean_microcat_embed = get_mean_last_microcats_embed(
                [microcat_to_index[elem] for elem in last_interactions if elem != -1],
                item_vectors,
            )
            for i, elem in enumerate(mean_microcat_embed):
                info[f"last_microcat_embed_value_{i + 1}"].append(elem)

            # вектор пользователя
            user_embed = get_user_embed(user_to_index[user_id], user_vectors)
            for i, elem in enumerate(user_embed):
                info[f"user_embed_value_{i + 1}"].append(elem)

            # вектор таргет микрокатегории
            item_embed = get_microcat_embed(new_negative_microcat, item_vectors)
            for i, elem in enumerate(item_embed):
                info[f"item_embed_value_{i + 1}"].append(elem)

            info["interaction_type"].append(str(random.sample(all_eids, 1)[0]))
            info["target_microcat"].append(str(new_negative_microcat))
            info["target_score"].append(
                get_als_score(
                    user_to_index[user_id],
                    new_negative_microcat,
                    user_vectors,
                    item_vectors,
                )
            )
            info["target"].append(0)

    return pd.DataFrame(info)


def get_catboost_inference_data(
    frame,
    user_to_index,
    microcat_to_index,
    user_vectors,
    item_vectors,
    last_interactions_prod,
):

    info = defaultdict(list)
    for i, row in frame.iterrows():

        for als_microcat_pred in row.preds:

            info["user_id"].append(str(row.user_id))
            info["target_microcat"].append(str(als_microcat_pred))
            info["target_score"].append(
                get_als_score(
                    user_to_index[row.user_id],
                    als_microcat_pred,
                    user_vectors,
                    item_vectors,
                )
            )

            # последние взаимодействия по микрокатегориям
            last_interactions = get_last_interactions(
                row.user_id, last_interactions_prod
            )
            for i, elem in enumerate(
                list(map(lambda x: str(microcat_to_index[x]), last_interactions))
            ):
                info[f"last_microcat_{i + 1}"].append(elem)

            # усредненный вектор последних 5 микрокатегорий
            mean_microcat_embed = get_mean_last_microcats_embed(
                [microcat_to_index[elem] for elem in last_interactions if elem != -1],
                item_vectors,
            )
            for i, elem in enumerate(mean_microcat_embed):
                info[f"last_microcat_embed_value_{i + 1}"].append(elem)

            # вектор пользователя
            user_embed = get_user_embed(user_to_index[row.user_id], user_vectors)
            for i, elem in enumerate(user_embed):
                info[f"user_embed_value_{i + 1}"].append(elem)

            # вектор таргет микрокатегории
            item_embed = get_microcat_embed(als_microcat_pred, item_vectors)
            for i, elem in enumerate(item_embed):
                info[f"item_embed_value_{i + 1}"].append(elem)

    return pd.DataFrame(info)


# Сортирует микрокатегории в порядке обратном скорам
def sort_preds(tdf, model_col):

    microcats = tdf.target_microcat.tolist()
    model_scores = tdf[model_col].tolist()

    pairs_model = list(zip(microcats, model_scores))
    pairs_model_sorted = sorted(pairs_model, key=lambda x: x[1], reverse=True)

    answers_model = [int(k) for (k, _) in pairs_model_sorted]

    return pd.Series({model_col: answers_model})


def create_mappings(
    interactions: DataFrame,
    user_to_index: Dict[int, int] = dict(),
    microcat_to_index: Dict[int, int] = dict(),
    save_path: str | Path = "",
) -> Tuple[DataFrame, Dict[int, int], Dict[int, int]]:
    if len(user_to_index) == 0:
        user_ids = interactions["user_id"].unique()
        user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

    if len(microcat_to_index) == 0:
        microcat_ids = interactions["microcat_id"].unique()
        microcat_to_index = {
            microcat_id: idx for idx, microcat_id in enumerate(microcat_ids)
        }

    interactions["user_idx"] = interactions["user_id"].map(user_to_index)
    interactions["microcat_idx"] = interactions["microcat_id"].map(microcat_to_index)

    if save_path != "":
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "user_to_index": user_to_index,
                    "microcat_to_index": microcat_to_index,
                    "index_to_user": {v: k for k, v in user_to_index.items()},
                    "index_to_item": {v: k for k, v in microcat_to_index.items()},
                },
                f,
            )

    return interactions, user_to_index, microcat_to_index


def save_csr_matrix(
    train_matrix: csr_matrix,
    eval_matrix: csr_matrix,
    save_path_train: str | Path,
    save_path_eval: str | Path,
) -> None:
    with open(save_path_train, "wb") as f:
        pickle.dump(train_matrix, f)

    with open(save_path_eval, "wb") as f:
        pickle.dump(eval_matrix, f)

    return None
