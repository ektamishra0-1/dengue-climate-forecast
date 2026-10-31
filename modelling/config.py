"""
Configuration module for classical time-series forecasting.
Specifies project paths, data partitions, target variables,
and representative microregions for Phase 2.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dengue_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Ensure all result directories exist
for directory in [FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Chronological split definitions (strict time boundary, no shuffling)
TRAIN_START = "2002-01-01"
TRAIN_END = "2015-12-01"

VAL_START = "2016-01-01"
VAL_END = "2017-12-01"

TEST_START = "2018-01-01"
TEST_END = "2019-12-01"

# Target variable specifications
PRIMARY_TARGET = "log1p_incidence"
SECONDARY_TARGET = "cases_per_100k"
RAW_CASE_TARGET = "dengue_cases"

# Exogenous climate variables for SARIMAX extension
# Informed by empirical lag cross-correlations in EDA (1 to 2 month thermal/moisture leads)
EXOG_VARS = ["tmax_lag_2", "tmin_lag_1", "pdsi_lag_2"]

# Representative Microregion Selection
# Documented Selection Rule:
# Selected exactly one prominent microregion from each of the 5 Brazilian macro-regions,
# capturing diverse biomes, population scales, and endemicity dynamics:
# 1. 31030 (Belo Horizonte, MG) - Southeast, Atlantic Rainforest, metropolitan explosive outbreaks
# 2. 52010 (Goiânia, GO) - Centre-West, Cerrado savanna, hyper-endemic baseline
# 3. 23016 (Fortaleza, CE) - Northeast, Caatinga semi-arid coastal, acute seasonal peaks
# 4. 13007 (Manaus, AM) - North, Amazon equatorial basin, high-humidity transmission
# 5. 41009 (Maringá, PR) - South, Atlantic Rainforest, subtropical with winter suppression
REPRESENTATIVE_MICROREGIONS = {
    31030: {
        "name": "Belo Horizonte",
        "clean_name": "Belo Horizonte",
        "state": "Minas Gerais",
        "region": "Southeast",
        "biome": "Atlantic Rainforest",
        "description": "Metropolitan epidemic hub in the Southeast with explosive multi-year peaks."
    },
    52010: {
        "name": "Goiânia",
        "clean_name": "Goiania",
        "state": "Goiás",
        "region": "Centre-West",
        "biome": "Cerrado",
        "description": "Hyper-endemic Cerrado savanna with high year-round baseline transmission."
    },
    23016: {
        "name": "Fortaleza",
        "clean_name": "Fortaleza",
        "state": "Ceará",
        "region": "Northeast",
        "biome": "Caatinga",
        "description": "Semi-arid coastal metropolis subject to severe seasonal epidemics."
    },
    13007: {
        "name": "Manaus",
        "clean_name": "Manaus",
        "state": "Amazonas",
        "region": "North",
        "biome": "Amazon",
        "description": "Equatorial Amazon rainforest basin with precipitation-driven seasonality."
    },
    41009: {
        "name": "Maringá",
        "clean_name": "Maringa",
        "state": "Paraná",
        "region": "South",
        "biome": "Atlantic Rainforest",
        "description": "Subtropical southern agricultural/urban center with strong winter lull."
    }
}

# National aggregate code identifier
NATIONAL_CODE = 0
NATIONAL_METADATA = {
    "name": "National Aggregate (Brazil)",
    "clean_name": "National_Aggregate",
    "state": "Brazil",
    "region": "National",
    "biome": "Continental Brazil",
    "description": "Monthly national aggregate summing dengue cases and population across all 558 microregions."
}

# Random seed for computational reproducibility
RANDOM_SEED = 42

# Seasonal period
SEASONAL_PERIOD = 12

