"""
Unit tests for evaluation metrics and zero-handling logic.
"""
import pytest
import numpy as np
import pandas as pd
from modelling.evaluation import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    symmetric_mape,
    mean_absolute_scaled_error,
    evaluate_forecast
)


def test_standard_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])
    
    mae = mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 2.25)
    
    rmse = root_mean_squared_error(y_true, y_pred)
    assert rmse > mae


def test_zero_handling_in_mape():
    # If actual observations contain zeros, MAPE should handle them cleanly
    y_true = np.array([0.0, 0.0, 50.0, 100.0])
    y_pred = np.array([2.0, 1.0, 45.0, 110.0])
    
    res = mean_absolute_percentage_error(y_true, y_pred, threshold=1.0)
    # Only 50 and 100 should be evaluated
    assert res["mape_evaluated_pct"] == 50.0
    assert res["mape_excluded_count"] == 2
    # Errors for positive: |50-45|/50 = 0.10, |100-110|/100 = 0.10 -> 10%
    assert np.isclose(res["mape"], 10.0)


def test_smape_bounded():
    # sMAPE must remain strictly bounded between 0 and 200%
    y_true = np.array([0.0, 10.0, 50.0])
    y_pred = np.array([5.0, 0.0, 50.0])
    
    smape_val = symmetric_mape(y_true, y_pred)
    assert 0.0 <= smape_val <= 200.0


def test_mase_calculation():
    y_train = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 11, 13, 15, 17])
    y_true = np.array([18.0, 20.0])
    y_pred = np.array([18.0, 20.0])  # perfect forecast
    
    mase_val = mean_absolute_scaled_error(y_true, y_pred, y_train, seasonal_period=12)
    assert np.isclose(mase_val, 0.0)
