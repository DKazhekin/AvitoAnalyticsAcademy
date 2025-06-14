import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import implicit
from scipy.sparse import csr_matrix


class ALS:
    def __init__(
        self,
        factors: int = 100,
        iterations: int = 10,
        regularization: float = 0.05,
        alpha: float = 1.0,
        random_state: int = 42,
        model: implicit.als.AlternatingLeastSquares = None,
    ):
        self.model = model
        if model is None:
            self.model = implicit.als.AlternatingLeastSquares(
                factors=factors,
                iterations=iterations,
                alpha=alpha,
                regularization=regularization,
                random_state=random_state,
            )

        self.mapping: Dict[str, Dict[int, int]] = dict()
        self.train_matrix: csr_matrix = None

    def train(
        self,
        train_matrix: csr_matrix,
        show_progress: bool = False,
        save_path: str = "",
        save_matrix: bool = False,
    ):
        self.model.fit(train_matrix, show_progress=show_progress)
        if save_path != "":
            with open(save_path, "wb") as f:
                pickle.dump(self.model, f)

        if save_matrix:
            self.train_matrix = train_matrix

    def load_train_matrix(self, train_matrix: csr_matrix | str | Path) -> None:
        self.train_matrix = self._train_matrix(train_matrix)
        return None

    def _train_matrix(self, train_matrix: csr_matrix | str | Path) -> csr_matrix:
        assert (
            isinstance(train_matrix, str)
            or isinstance(train_matrix, csr_matrix)
            or isinstance(train_matrix, Path)
        )

        if isinstance(train_matrix, str) or isinstance(train_matrix, Path):
            with open(train_matrix, "rb") as f:
                train_matrix = pickle.load(f)
        elif isinstance(train_matrix, csr_matrix):
            train_matrix = train_matrix

        return train_matrix

    def load_mapping(self, mapping: Dict[str, Dict[int, int]] | str | Path) -> None:
        self.mapping = self._mapping(mapping)
        return None

    def _mapping(
        self, mapping: Dict[str, Dict[int, int]] | str | Path
    ) -> Dict[str, Dict[int, int]]:
        assert (
            isinstance(mapping, str)
            or isinstance(mapping, dict)
            or isinstance(mapping, Path)
        )

        if isinstance(mapping, str) or isinstance(mapping, Path):
            with open(mapping, "rb") as f:
                m: Dict[str, Dict[int, int]] = pickle.load(f)
        elif isinstance(mapping, dict):
            m = mapping

        return m

    def get_similar_items(
        self,
        item_id: int,
        N: int = 5,
        verbose: bool = False,
    ) -> List[Tuple[int, float]]:
        assert len(self.mapping) != 0

        item_to_index = self.mapping["item_to_index"]
        index_to_item = self.mapping["index_to_item"]

        if item_id not in item_to_index:
            if verbose:
                print(f"Товар {item_id} отсутствует в обучающих данных")
            return []

        item_idx = item_to_index[item_id]

        if item_idx >= self.model.item_factors.shape[0]:
            if verbose:
                print(f"Товарный индекс {item_idx} выходит за пределы модели")
            return []

        similar = self.model.similar_items(item_idx, N=N + 1)

        return [
            (index_to_item[idx], score)
            for idx, score in zip(similar[0], similar[1])
            if idx != item_idx
        ][:N]

    def get_user_recommendations(
        self,
        user_id: int,
        train_matrix: csr_matrix = None,
        N: int = 10,
        filter_viewed: bool = True,
        verbose: bool = False,
    ) -> List[Tuple[int, float]]:
        assert len(self.mapping) != 0
        assert self.train_matrix is not None or train_matrix is not None

        if train_matrix is None:
            train_matrix = self.train_matrix

        user_to_index = self.mapping["user_to_index"]
        index_to_item = self.mapping["index_to_item"]

        if user_id not in user_to_index:
            if verbose:
                print(f"Пользователь {user_id} отсутствует в обучающих данных")
            return []

        user_idx = user_to_index[user_id]

        recommendations = self.model.recommend(
            user_idx,
            train_matrix[user_idx],
            N=N,
            filter_already_liked_items=filter_viewed,
        )

        return [
            (index_to_item[idx], score)
            for idx, score in zip(recommendations[0], recommendations[1])
        ]

    def get_similar_users(
        self, user_id: int, N: int = 5, verbose: bool = False
    ) -> List[Tuple[int, float]]:
        assert len(self.mapping) != 0

        if user_id not in self.mapping["user_to_index"]:
            if verbose:
                print(f"Пользователь {user_id} отсутствует в обучающих данных")
            return []

        user_idx = self.mapping["user_to_index"][user_id]
        similar = self.model.similar_users(user_idx, N=N + 1)

        return [
            (self.mapping["index_to_user"][idx], score)
            for idx, score in zip(similar[0], similar[1])
            if idx != user_idx
        ][:N]

    def get_user_recommendations_with_similar_users(
        self,
        user_id: int,
        train_matrix: csr_matrix = None,
        N: int = 10,
        similar_users_weight: float = 1.3,
        verbose: bool = False,
    ) -> List[Tuple[int, float]]:
        assert len(self.mapping) != 0
        assert self.train_matrix is not None or train_matrix is not None

        if train_matrix is None:
            train_matrix = self.train_matrix

        main_rec = self.get_user_recommendations(
            user_id, train_matrix, N=N * 2, verbose=verbose
        )

        similar_users = self.get_similar_users(user_id, N=3, verbose=verbose)

        similar_rec = []
        for similar_user_id, _ in similar_users:
            similar_rec.extend(
                self.get_user_recommendations(
                    similar_user_id, train_matrix, N=N, verbose=verbose
                )
            )

        combined = main_rec + [
            (item, score * similar_users_weight) for item, score in similar_rec
        ]
        return sorted(combined, key=lambda x: -x[1])[:N]

    @staticmethod
    def load_pickle(path: str):
        with open(path, "rb") as f:
            model = pickle.load(f)

        return ALS(model=model)
