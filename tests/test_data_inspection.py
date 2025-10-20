# tests/test_data_inspection.py
import pandas as pd
import pytest

from src.utils.data_inspection import invalids_summary

def make_df():
    # row0: valid
    r0 = dict(Open=10, High=11, Low=9, Close=10.5, Adj_Close=10.5, Volume=100)
    # exact duplicate of row0 -> counts as 1 duplicate (the second occurrence)
    r1 = r0.copy()
    # row2: all price fields <= 0 (invalid), volume positive
    r2 = dict(Open=0, High=0, Low=0, Close=0, Adj_Close=0, Volume=5)
    # row3: negative volume, otherwise valid
    r3 = dict(Open=20, High=21, Low=19, Close=20.5, Adj_Close=20.5, Volume=-3)

    # Note: your code expects column name "Adj Close" (with a space). Build with that.
    df = pd.DataFrame([r0, r1, r2, r3]).rename(columns={"Adj_Close": "Adj Close"})
    return df

def test_invalids_summary_counts():
    df = make_df()
    res = invalids_summary(df)

    # one row labeled TOTAL
    assert list(res.index) == ["TOTAL"]
    assert res.shape == (1, 7)

    # expected counts:
    # - Negative volumes: only row3 -> 1
    # - Each price <= 0: only row2 -> 1 each
    # - Duplicates: row1 is duplicate of row0 -> 1
    expected = {
        "Negative volumes": 1,
        "Open price <= 0": 1,
        "Close price <= 0": 1,
        "Adj Close <= 0": 1,
        "High <= 0": 1,
        "Low <= 0": 1,
        "Duplicates": 1,
    }

    # column presence + exact values
    assert set(res.columns) == set(expected.keys())
    for k, v in expected.items():
        assert int(res.loc["TOTAL", k]) == v

def test_missing_required_columns_raises():
    # drop a required column -> should raise ValueError
    df = make_df().drop(columns=["Adj Close"])
    with pytest.raises(ValueError):
        invalids_summary(df)

def test_extra_columns_ok():
    # adding extra columns should not affect the result
    df = make_df()
    df["Extra"] = [1, 2, 3, 4]
    res = invalids_summary(df)
    assert "Extra" not in res.columns
