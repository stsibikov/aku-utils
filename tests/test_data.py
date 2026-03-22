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
