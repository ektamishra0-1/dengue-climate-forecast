"""
Data loader module for DengueCast Streamlit dashboard.
Provides cached, leak-free retrieval of precomputed EDA datasets,
stationarity audits, evaluation metrics, and prediction files.
"""
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dengue_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

LOCATION_MAP = {
    "Belo Horizonte": {
        "code": 31030,
        "clean_name": "Belo Horizonte",
        "display_name": "Belo Horizonte (Southeast, Atlantic Rainforest)",
        "state": "Minas Gerais",
        "region": "Southeast",
        "biome": "Atlantic Rainforest",
        "desc": "Metropolitan epidemic hub in the Southeast with explosive multi-year outbreak cycles."
    },
    "Goiania": {
        "code": 52010,
        "clean_name": "Goiania",
        "display_name": "Goiânia (Centre-West, Cerrado)",
        "state": "Goiás",
        "region": "Centre-West",
        "biome": "Cerrado",
        "desc": "Hyper-endemic Cerrado savanna hub with high year-round baseline transmission."
    },
    "Fortaleza": {
        "code": 23016,
        "clean_name": "Fortaleza",
        "display_name": "Fortaleza (Northeast, Caatinga)",
        "state": "Ceará",
        "region": "Northeast",
        "biome": "Caatinga",
        "desc": "Equatorial semi-arid coastal metropolis prone to severe, recurring seasonal outbreaks."
    },
    "Manaus": {
        "code": 13007,
        "clean_name": "Manaus",
        "display_name": "Manaus (North, Amazon)",
        "state": "Amazonas",
        "region": "North",
        "biome": "Amazon",
        "desc": "Equatorial Amazon rainforest basin with high humidity and precipitation-driven dynamics."
    },
    "Maringa": {
        "code": 41009,
        "clean_name": "Maringa",
        "display_name": "Maringá (South, Atlantic Rainforest)",
        "state": "Paraná",
        "region": "South",
        "biome": "Atlantic Rainforest",
        "desc": "Subtropical southern agricultural/urban center with strong winter cold suppression."
    },
    "National_Aggregate": {
        "code": 0,
        "clean_name": "National_Aggregate",
        "display_name": "National Aggregate (Continental Brazil)",
        "state": "Brazil",
        "region": "National",
        "biome": "Continental Brazil",
        "desc": "Continental macroeconomic benchmark summing all 558 microregions across Brazil."
    }
}


@st.cache_data(show_spinner=False)
def load_eda_dataset() -> pd.DataFrame:
    """
    Load essential columns from the master feature dataset for EDA and climate analytics.
    """
    cols = [
        "date", "micro_code", "micro_name", "state_name", "region_name", "biome_name",
        "month", "year", "dengue_cases", "population", "cases_per_100k", "log1p_incidence",
        "tmax", "tmin", "dtr", "pdsi", "urban", "water_network", "water_shortage"
    ]
    df = pd.read_csv(DATA_PATH, usecols=cols, parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_national_timeseries() -> pd.DataFrame:
    """
    Precompute the monthly continental Brazil national time series.
    """
    df = load_eda_dataset()
    nat = df.groupby("date").agg(
        total_cases=("dengue_cases", "sum"),
        total_pop=("population", "sum"),
        mean_tmax=("tmax", "mean"),
        mean_tmin=("tmin", "mean"),
        mean_dtr=("dtr", "mean"),
        mean_pdsi=("pdsi", "mean")
    ).reset_index()
    
    # Population is not available for burn-in year 2000; safely handle zero pop
    nat["cases_per_100k"] = np.where(
        nat["total_pop"] > 0,
        (nat["total_cases"] / nat["total_pop"]) * 100_000.0,
        0.0
    )
    nat["log1p_incidence"] = np.log1p(nat["cases_per_100k"])
    nat["year"] = nat["date"].dt.year
    nat["month"] = nat["date"].dt.month
    return nat


@st.cache_data(show_spinner=False)
def load_all_metrics() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the stationarity audit and all validation and test model metric tables.
    """
    stat_df = pd.read_csv(METRICS_DIR / "stationarity_differencing_audit.csv")
    val_df = pd.read_csv(METRICS_DIR / "validation_metrics_all_models.csv")
    test_df = pd.read_csv(METRICS_DIR / "test_metrics_all_models.csv")
    return stat_df, val_df, test_df


@st.cache_data(show_spinner=False)
def load_prediction_series(period: str, clean_name: str) -> Optional[pd.DataFrame]:
    """
    Load the precomputed 1-step rolling out-of-sample prediction file.
    period: 'val' or 'test'
    """
    file_path = PREDICTIONS_DIR / f"predictions_{period}_{clean_name}.csv"
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df
