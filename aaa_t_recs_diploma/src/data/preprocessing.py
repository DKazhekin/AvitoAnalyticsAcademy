import datetime
from collections import defaultdict

from pandas import DataFrame
from sklearn.model_selection import train_test_split


def add_session_ids_df(df: DataFrame, session_interval_hours: float) -> None:
    assert "user_id" in df.columns, "нету user_id в датафрейме"
    assert "event_date" in df.columns, "нету event_date в датафрейме"

    session_interval = datetime.timedelta(hours=session_interval_hours)
    df.sort_values(by=["user_id", "event_date"], inplace=True)
    df["event_diff"] = df.groupby("user_id")["event_date"].diff()
    df["session_id"] = (
        (df["event_diff"].isna()) | (df["event_diff"] > session_interval)
    ).cumsum()
    df.drop(columns=["event_diff"], inplace=True)

    return None


def train_test_split_stratify(X, test_train_ratio: float):
    assert test_train_ratio > 0 and test_train_ratio < 1

    item_to_sessions = defaultdict(list)
    for idx, session in enumerate(X["item_id"]):
        for item in session:
            item_to_sessions[item].append(idx)

    train_indices = []
    test_indices = []

    for item, sessions in item_to_sessions.items():
        train_item, test_item = train_test_split(
            sessions,
            test_size=test_train_ratio,
            random_state=42,
        )
        train_indices.extend(train_item)
        test_indices.extend(test_item)

    train_indices = list(set(train_indices))
    test_indices = list(set(test_indices))

    train_sessions = X.iloc[train_indices]
    test_sessions = X.iloc[test_indices]

    return train_sessions, test_sessions


def prepare_df_min_len_count(
    df: DataFrame, min_len_session: int, min_count_item: int, verbose: bool = True
):
    assert "item_id" in df.columns
    assert "session_id" in df.columns

    df.dropna(inplace=True, subset=["item_id", "session_id"], how="any")
    prepare_train_buyer_stream = df.groupby(by=["session_id"])["item_id"].apply(set)
    prepare_train_buyer_stream = prepare_train_buyer_stream[
        prepare_train_buyer_stream.apply(len) >= min_len_session
    ]
    prepare_train_buyer_stream = prepare_train_buyer_stream.explode().reset_index()
    len_df = -1

    i = 0
    while len_df != len(prepare_train_buyer_stream) and len_df != 0:
        len_df = len(prepare_train_buyer_stream)
        if verbose:
            print(f"Итерация {i}: {len_df} записей")

        buyer_stream_len_ses = prepare_train_buyer_stream.groupby(by=["session_id"])[
            "item_id"
        ].count()
        buyer_stream_len_ses = buyer_stream_len_ses[
            buyer_stream_len_ses >= min_len_session
        ]
        prepare_train_buyer_stream = prepare_train_buyer_stream[
            prepare_train_buyer_stream["session_id"].isin(buyer_stream_len_ses.index)
        ]

        buyer_stream_item_count = prepare_train_buyer_stream["item_id"].value_counts()
        buyer_stream_item_count = buyer_stream_item_count[
            buyer_stream_item_count >= min_count_item
        ]
        prepare_train_buyer_stream = prepare_train_buyer_stream[
            prepare_train_buyer_stream["item_id"].isin(buyer_stream_item_count.index)
        ]
        i += 1

    prepare_train_buyer_stream = (
        prepare_train_buyer_stream.groupby(by=["session_id"])["item_id"]
        .apply(set)
        .reset_index()
    )

    return prepare_train_buyer_stream
