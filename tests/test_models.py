"""
Unit tests for forecasting models on synthetic monthly series.
"""
import pytest
import numpy as np
import pandas as pd
from modelling.models import (
    SeasonalNaiveForecaster,
    ARIMAForecaster,
    SARIMAForecaster,
    HoltWintersForecaster
)


@pytest.fixture
def synthetic_series():
    np.random.seed(42)
    dates = pd.date_range("2005-01-01", periods=60, freq="MS")
    # Trend + 12-month seasonality + noise
    t = np.arange(60)
    seasonal = 2.0 * np.sin(2 * np.pi * t / 12.0)
    trend = 0.05 * t
    y = pd.Series(5.0 + trend + seasonal + np.random.normal(0, 0.2, 60), index=dates)
    return y


def test_seasonal_naive(synthetic_series):
    train = synthetic_series.iloc[:48]
    val = synthetic_series.iloc[48:]
    
    model = SeasonalNaiveForecaster(seasonal_period=12)
    model.fit(train)
    preds = model.predict_one_step(synthetic_series, val.index)
    
    assert len(preds) == len(val)
    # Check that pred at val.index[0] matches observation 12 months prior
    expected = synthetic_series.loc[val.index[0] - pd.DateOffset(months=12)]
    assert np.isclose(preds.iloc[0], expected)


def test_arima_forecaster(synthetic_series):
    train = synthetic_series.iloc[:48]
    val = synthetic_series.iloc[48:]
    
    model = ARIMAForecaster(order=(1, 1, 0))
    model.fit(train)
    preds = model.predict_one_step(synthetic_series, val.index)
    
    assert len(preds) == len(val)
    assert not preds.isna().any()


def test_sarima_forecaster(synthetic_series):
    train = synthetic_series.iloc[:48]
    val = synthetic_series.iloc[48:]
    
    model = SARIMAForecaster(order=(1, 0, 0), seasonal_order=(1, 1, 0, 12))
    model.fit(train)
    preds = model.predict_one_step(synthetic_series, val.index)
    
    assert len(preds) == len(val)
    assert not preds.isna().any()


def test_holtwinters_forecaster(synthetic_series):
    train = synthetic_series.iloc[:48]
    val = synthetic_series.iloc[48:]
    
    model = HoltWintersForecaster(trend="add", seasonal="add", seasonal_periods=12)
    model.fit(train)
    preds = model.predict_one_step(synthetic_series, val.index)
    
    assert len(preds) == len(val)
    assert not preds.isna().any()

