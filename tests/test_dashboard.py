"""
Unit tests for DengueCast Streamlit dashboard data loader,
metric consistency, and prediction file validation.
"""
import pytest
import numpy as np
import pandas as pd
from dashboard.data_loader import (
    load_eda_dataset,
    load_national_timeseries,
    load_all_metrics,
    load_prediction_series,
    LOCATION_MAP,
    METRICS_DIR,
    PREDICTIONS_DIR
)


def test_load_eda_dataset_schema():
    df = load_eda_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 133920  # 558 microregions * 240 months
    
    required_cols = [
        "date", "micro_code", "micro_name", "state_name", "region_name", "biome_name",
        "month", "year", "dengue_cases", "population", "cases_per_100k", "log1p_incidence",
        "tmax", "tmin", "dtr", "pdsi"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
        
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["log1p_incidence"].min() >= 0.0


def test_load_national_timeseries():
    nat = load_national_timeseries()
    assert isinstance(nat, pd.DataFrame)
    assert len(nat) == 240  # 240 months from 2000 to 2019
    assert nat["date"].min() == pd.Timestamp("2000-01-01")
    assert nat["date"].max() == pd.Timestamp("2019-12-01")
    assert nat["total_cases"].sum() > 0
    assert nat["cases_per_100k"].min() >= 0.0
    assert not nat["log1p_incidence"].isna().any()


def test_load_all_metrics():
    stat_df, val_df, test_df = load_all_metrics()
    
    # Stationarity audit checks
    assert len(stat_df) >= 6
    assert "Location" in stat_df.columns
    assert "ADF p-val" in stat_df.columns
    assert "KPSS p-val" in stat_df.columns
    
    # Metrics tables checks
    for m_df, name in [(val_df, "Validation"), (test_df, "Test")]:
        assert isinstance(m_df, pd.DataFrame)
        assert len(m_df) >= 30  # 6 locations * 5 models
        expected_cols = ["Location", "Model", "MAE (log1p)", "RMSE (log1p)", "sMAPE% (log1p)", "MASE (log1p)"]
        for col in expected_cols:
            assert col in m_df.columns, f"Missing {col} in {name} metrics"
        assert (m_df["MAE (log1p)"] >= 0).all()
        assert (m_df["RMSE (log1p)"] >= 0).all()


def test_predictions_files_integrity():
    required_models = ["Seasonal_Naive", "ARIMA", "SARIMA", "Holt_Winters", "SARIMAX_Climate"]
    
    for loc_key, loc_info in LOCATION_MAP.items():
        clean_name = loc_info["clean_name"]
        
        # Test Validation predictions
        df_val = load_prediction_series("val", clean_name)
        assert df_val is not None, f"Missing validation prediction file for {clean_name}"
        assert len(df_val) == 24, f"Validation period must have 24 months for {clean_name}"
        assert df_val.index.min() == pd.Timestamp("2016-01-01")
        assert df_val.index.max() == pd.Timestamp("2017-12-01")
        for m in required_models:
            assert m in df_val.columns, f"Missing model {m} in val predictions for {clean_name}"
        assert "actual_log1p" in df_val.columns
        assert "actual_orig" in df_val.columns
        
        # Test Test predictions
        df_test = load_prediction_series("test", clean_name)
        assert df_test is not None, f"Missing test prediction file for {clean_name}"
        assert len(df_test) == 24, f"Test period must have 24 months for {clean_name}"
        assert df_test.index.min() == pd.Timestamp("2018-01-01")
        assert df_test.index.max() == pd.Timestamp("2019-12-01")
        for m in required_models:
            assert m in df_test.columns, f"Missing model {m} in test predictions for {clean_name}"
        assert "actual_log1p" in df_test.columns
        assert "actual_orig" in df_test.columns


def test_metrics_and_predictions_consistency():
    """Verify that predictions stored in CSV match the reported MAE metric in test_metrics_all_models.csv."""
    _, _, test_df = load_all_metrics()
    
    # Check SARIMA on National Aggregate
    nat_preds = load_prediction_series("test", "National_Aggregate")
    actual = nat_preds["actual_log1p"].values
    sarima_pred = nat_preds["SARIMA"].values
    
    calc_mae = float(np.mean(np.abs(actual - sarima_pred)))
    
    row = test_df[(test_df["Location"] == "National Aggregate (Brazil)") & (test_df["Model"] == "SARIMA")]
    assert not row.empty
    reported_mae = float(row["MAE (log1p)"].iloc[0])
    
    # Tolerant to 3 decimal places
    assert np.isclose(calc_mae, reported_mae, atol=1e-3)


def test_streamlit_app_all_sections_end_to_end():
    """Verify that all 6 navigation sections render without raising any runtime exceptions."""
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    
    # 1. Overview
    at.run()
    assert not at.exception, f"Overview section raised exception: {at.exception}"
    
    # 2. EDA
    at.sidebar.radio[0].set_value("2. Exploratory Data Analysis").run()
    assert not at.exception, f"EDA section raised exception: {at.exception}"
    
    # 3. Climate
    at.sidebar.radio[0].set_value("3. Climate Analysis").run()
    assert not at.exception, f"Climate section raised exception: {at.exception}"
    
    # 4. Stationarity
    at.sidebar.radio[0].set_value("4. Stationarity & Diagnostics").run()
    assert not at.exception, f"Stationarity section raised exception: {at.exception}"
    
    # 5. Model Comparison
    at.sidebar.radio[0].set_value("5. Model Comparison").run()
    assert not at.exception, f"Model comparison section raised exception: {at.exception}"
    
    # 6. Forecast Explorer
    at.sidebar.radio[0].set_value("6. Forecast Explorer").run()
    assert not at.exception, f"Forecast explorer section raised exception: {at.exception}"

