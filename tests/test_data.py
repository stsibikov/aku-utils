import numpy as np
import pandas as pd
import pytest


class TestDrop:
    @pytest.mark.parametrize("drop_cols, expected_cols", [(["a"], ["b", "c"]), (["b", "c"], ["a"])])
    def test_all_cols_there(self, drop_cols, expected_cols):
        from aku_utils.data import drop

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1, 2, 3],
            "c": [1, 2, 3],
        })
        df = drop(df, drop_cols, if_no_column="raise")
        assert list(df.columns) == expected_cols

    @pytest.mark.parametrize(
        "drop_cols, expected_cols", [(["a", "d"], ["b", "c"]), (["e"], ["a", "b", "c"])]
    )
    def test_extra_cols(self, drop_cols, expected_cols):
        from aku_utils.data import drop

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [1, 2, 3],
            "c": [1, 2, 3],
        })
        df = drop(df, drop_cols, if_no_column="skip")
        assert list(df.columns) == expected_cols


class TestCoalesce:
    @pytest.mark.parametrize(
        "cols, value, expected_data",
        [
            (["a", "b", "c", "d"], None, [1, 3, 5, np.nan]),
            (["b", "c"], None, [1, 3, 6, np.nan]),
            (["a", "b", "c", "d"], 0, [1, 3, 5, 0]),
        ],
    )
    def test(self, cols, value, expected_data):
        import string

        from aku_utils.data import coalesce

        data = []
        data.append([np.nan, 1, np.nan, np.nan])
        data.append([np.nan, np.nan, 3, 4])
        data.append([5, 6, np.nan, 7])
        data.append([np.nan, np.nan, np.nan, np.nan])

        df = pd.DataFrame(data, columns=list(string.ascii_lowercase[: len(data[0])]))

        srs = coalesce(df, cols, value=value)
        assert srs.equals(pd.Series(expected_data, dtype="float"))


class TestReorder:
    @pytest.mark.parametrize(
        "value, expected_cols",
        [
            (["b", "c"], ["b", "c", "a", "d"]),
            (["c"], ["c", "a", "b", "d"]),
            ({"b": 0}, ["b", "a", "c", "d"]),
            ({"b": 1, "c": 0}, ["c", "b", "a", "d"]),
            ({"d": 0, "c": 2}, ["d", "a", "c", "b"]),
        ],
    )
    def test(self, value, expected_cols):
        from aku_utils.data import reorder

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]})

        df = reorder(df, value)
        assert list(df.columns) == expected_cols

    def test_same_data(self):
        from aku_utils.data import reorder

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [10, 11, 12]})

        df = reorder(df, ["b", "c"])
        assert df["a"].equals(pd.Series([1, 2, 3]))
        assert df["b"].equals(pd.Series([4, 5, 6]))


class TestDedupe:
    @pytest.mark.parametrize(
        "cols, expected_data",
        [
            (
                ["a", "b"],
                [
                    (0, 1, 2, 3),
                    (4, 5, 6, 7),
                    (8, 9, 10, 11),
                    (8, 10, 10, 12),
                    (12, 13, 14, 15),
                ],
            ),
            (
                ["a"],
                [
                    (0, 1, 2, 3),
                    (4, 5, 6, 7),
                    (8, 9, 10, 11),
                    (12, 13, 14, 15),
                ],
            ),
        ],
    )
    def test(self, cols, expected_data):
        import string

        from aku_utils.data import dedupe

        data = []
        data.append([0, 1, 2, 3])
        data.append([4, 5, 6, 7])
        data.append([4, 5, 7, 8])
        data.append([8, 9, 10, 11])
        data.append([8, 10, 10, 12])
        data.append([12, 13, 14, 15])

        columns = list(string.ascii_lowercase[: len(data[0])])
        df = pd.DataFrame(data, columns=columns)

        df = dedupe(df, cols)

        test_df = pd.DataFrame(expected_data, columns=columns)
        assert test_df.equals(df.reset_index(drop=True))

    def test_return_dict(self, cols, expected_data):
        pass
