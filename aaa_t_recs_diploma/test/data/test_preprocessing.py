import pandas as pd
import pytest

from src.data import add_session_ids_df


def test_add_session_ids_df():
    df = pd.DataFrame({"user_id": [0] * 100})
    session_interval_hours = 3

    with pytest.raises(AssertionError) as excinfo:
        add_session_ids_df(df, session_interval_hours)

    assert "event_date" in str(
        excinfo.value
    ), "Ожидалась ошибка об отсутствии event_date"
