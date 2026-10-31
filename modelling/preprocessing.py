"""
Preprocessing module for classical time-series forecasting.
Handles data loading, regular monthly series extraction, national aggregation,
missing value auditing, chronological train/val/test splitting, and inverse scaling.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

from modelling.config import (
    DATA_PATH,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    PRIMARY_TARGET,
    SECONDARY_TARGET,
    EXOG_VARS
)


def load_feature_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the feature-engineered dataset and ensure correct chronological ordering.
    """
    path = filepath or DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {path.resolve()}")
    
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["micro_code", "date"]).reset_index(drop=True)
    return df


def extract_microregion_series(
    df: pd.DataFrame,
    micro_code: int,
    target_col: str = PRIMARY_TARGET,
    exog_cols: Optional[List[str]] = None,
    start_date: str = TRAIN_START,
    end_date: str = TEST_END
) -> Tuple[pd.Series, Optional[pd.DataFrame], pd.Series]:
    """
    Extract a regular monthly time series for a specific microregion.
    
    Returns:
        y_primary: pd.Series of primary target (log1p_incidence) with DatetimeIndex (freq='MS')
        exog_df: pd.DataFrame of exogenous variables (or None)
        y_secondary: pd.Series of original scale incidence (cases_per_100k)
    """
    sub = df[df["micro_code"] == micro_code].copy()
    if sub.empty:
        raise ValueError(f"Microregion code {micro_code} not found in dataset.")
    
    # Filter date range
    sub = sub[(sub["date"] >= start_date) & (sub["date"] <= end_date)]
    sub = sub.sort_values("date").set_index("date")
    
    # Enforce regular monthly beginning frequency
    sub = sub.asfreq("MS")
    
    # Check for missing values in target
    if sub[target_col].isna().any():
        n_missing = sub[target_col].isna().sum()
        raise ValueError(f"Target column {target_col} contains {n_missing} missing values in microregion {micro_code}.")
    
    y_primary = sub[target_col].rename(f"micro_{micro_code}_{target_col}")
    y_secondary = sub[SECONDARY_TARGET].rename(f"micro_{micro_code}_{SECONDARY_TARGET}")
    
    exog_df = None
    if exog_cols:
        # Check exogenous missing values
        exog_df = sub[exog_cols].copy()
        if exog_df.isna().any().any():
            # If Fernando de Noronha (26019) or similar has missing PDSI, interpolate with column mean/forward fill
            exog_df = exog_df.ffill().bfill()
    
    return y_primary, exog_df, y_secondary


def build_national_aggregate_series(
    df: pd.DataFrame,
    target_col: str = PRIMARY_TARGET,
    exog_cols: Optional[List[str]] = None,
    start_date: str = TRAIN_START,
    end_date: str = TEST_END
) -> Tuple[pd.Series, Optional[pd.DataFrame], pd.Series]:
    """
    Aggregate case counts and population across all 558 microregions to form
    a continental Brazil monthly time series.
    
    Important: National aggregation is an independent macro-level experiment
    and should not be conflated with microregion panel forecasting.
    
    Returns:
        y_primary: pd.Series of national log1p_incidence with freq='MS'
        exog_df: pd.DataFrame of national average climate features
        y_secondary: pd.Series of national incidence per 100k
    """
    sub = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    
    # Group by date: sum cases and population
    grouped = sub.groupby("date").agg(
        total_cases=("dengue_cases", "sum"),
        total_population=("population", "sum")
    ).reset_index()
    
    # Calculate national incidence per 100k
    grouped["cases_per_100k"] = (grouped["total_cases"] / grouped["total_population"]) * 100_000.0
    grouped["log1p_incidence"] = np.log1p(grouped["cases_per_100k"])
    
    grouped = grouped.sort_values("date").set_index("date").asfreq("MS")
    
    y_primary = grouped[target_col].rename(f"national_{target_col}")
    y_secondary = grouped[SECONDARY_TARGET].rename(f"national_{SECONDARY_TARGET}")
    
    exog_df = None
    if exog_cols:
        # Compute population-weighted or arithmetic mean of climate variables nationally
        climate_nat = sub.groupby("date")[exog_cols].mean().asfreq("MS")
        exog_df = climate_nat
    
    return y_primary, exog_df, y_secondary


def chronological_split(
    y: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END
) -> Dict[str, pd.Series]:
    """
    Split time-series strictly chronologically without random shuffling or data leakage.
    
    Partition Windows:
    - Train: start to 2015-12-01 (14 years = 168 months)
    - Validation: 2016-01-01 to 2017-12-01 (2 years = 24 months)
    - Test: 2018-01-01 to 2019-12-01 (2 years = 24 months)
    """
    y_train = y.loc[:train_end]
    y_val = y.loc[pd.Timestamp(train_end) + pd.DateOffset(months=1) : val_end]
    y_test = y.loc[pd.Timestamp(val_end) + pd.DateOffset(months=1) :]
    
    splits = {
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    
    if exog is not None:
        splits["exog_train"] = exog.loc[:train_end]
        splits["exog_val"] = exog.loc[pd.Timestamp(train_end) + pd.DateOffset(months=1) : val_end]
        splits["exog_test"] = exog.loc[pd.Timestamp(val_end) + pd.DateOffset(months=1) :]
    
    return splits


def inverse_log1p(y_log: np.ndarray | pd.Series | float) -> np.ndarray | pd.Series | float:
    """
    Transform predictions from log1p scale back to original incidence per 100k population:
    y_orig = max(0, expm1(y_log))
    """
    if isinstance(y_log, (pd.Series, pd.DataFrame)):
        return np.maximum(0.0, np.expm1(y_log))
    elif isinstance(y_log, np.ndarray):
        return np.maximum(0.0, np.expm1(y_log))
    else:
        return max(0.0, float(np.expm1(y_log)))

