"""
Unit tests for preprocessing and transformation logic.
"""
import pytest
import numpy as np
import pandas as pd
from modelling.preprocessing import (
    chronological_split, 
    inverse_log1p, 
    build_national_aggregate_series
)
from modelling.config import TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END


def test_inverse_log1p():
    # scalar test
    assert np.isclose(inverse_log1p(0.0), 0.0)
    assert np.isclose(inverse_log1p(np.log1p(100.0)), 100.0)
    assert inverse_log1p(-0.5) == 0.0  # clipped non-negative
    
    # series test
    s = pd.Series([0.0, np.log1p(50.0), np.log1p(200.0)])
    inv_s = inverse_log1p(s)
    assert np.allclose(inv_s.values, [0.0, 50.0, 200.0])


def test_chronological_split():
    dates = pd.date_range("2002-01-01", "2019-12-01", freq="MS")
    y = pd.Series(np.arange(len(dates)), index=dates)
    exog = pd.DataFrame({"temp": np.random.randn(len(dates))}, index=dates)
    
    splits = chronological_split(y, exog=exog)
    
    # Check bounds
    assert splits["y_train"].index[0] == pd.Timestamp("2002-01-01")
    assert splits["y_train"].index[-1] == pd.Timestamp("2015-12-01")
    assert len(splits["y_train"]) == 14 * 12  # 168 months
    
    assert splits["y_val"].index[0] == pd.Timestamp("2016-01-01")
    assert splits["y_val"].index[-1] == pd.Timestamp("2017-12-01")
    assert len(splits["y_val"]) == 2 * 12  # 24 months
    
    assert splits["y_test"].index[0] == pd.Timestamp("2018-01-01")
    assert splits["y_test"].index[-1] == pd.Timestamp("2019-12-01")
    assert len(splits["y_test"]) == 2 * 12  # 24 months
    
    # Check no overlap
    assert len(set(splits["y_train"].index).intersection(set(splits["y_val"].index))) == 0
    assert len(set(splits["y_val"].index).intersection(set(splits["y_test"].index))) == 0
    
    # Check exog splits
    assert len(splits["exog_train"]) == 168
    assert len(splits["exog_val"]) == 24
    assert len(splits["exog_test"]) == 24

