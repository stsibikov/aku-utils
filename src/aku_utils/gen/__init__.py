import string

import numpy as np
import pandas as pd


def dtypes_df(n_rows=1_000_000):
    start_date = np.datetime64("2020-01-01")
    end_date = np.datetime64("2024-01-01")
    days_range = (end_date - start_date).astype(int)

    data = {
        "date": start_date + np.random.randint(0, days_range, n_rows).astype("timedelta64[D]"),
        "dt": start_date
        + np.random.randint(0, days_range * 86400, n_rows).astype("timedelta64[s]"),
        "int": np.random.randint(0, 2**4, size=n_rows, dtype="int64"),
        "int_high": np.random.randint(0, 2**32, size=n_rows, dtype="int64"),
        "float": np.random.random(n_rows) * 100,
        "str": np.random.choice(list(string.ascii_lowercase), size=n_rows),
    }

    df = pd.DataFrame(data)
    df = df.reset_index(names=["id"])
    return df


def nan_df(n_rows=300_000, frac: float = 0.5):

    data = {
        "int": np.random.randint(0, 2**4, size=n_rows, dtype="int64"),
        "float0": np.random.random(n_rows) * 100,
        "float1": np.random.random(n_rows) * 100,
        "float2": np.random.random(n_rows) * 100,
    }

    df = pd.DataFrame(data)
    df = df.reset_index(names=["id"])

    for col in ["float0", "float1", "float2"]:
        index = df["id"].sample(frac=frac).index
        df.loc[index, col] = np.nan

    return df
