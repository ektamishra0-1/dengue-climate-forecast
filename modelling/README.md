# Phase 2 — Classical Time-Series Forecasting: Module Guide

This directory contains the complete implementation of **Phase 2 (Mid-Semester Evaluation)** of the Brazilian Dengue Climate Forecasting research project.

---

## 1. Directory Structure

```
modelling/
├── config.py                                   # Project paths, dates, targets, microregion metadata
├── preprocessing.py                            # Data loading, series extraction, national aggregation, splits
├── stationarity.py                             # ADF, KPSS, rolling statistics, differencing diagnostics
├── models.py                                   # Seasonal Naive, ARIMA, SARIMA, Holt-Winters, SARIMAX
├── evaluation.py                               # MAE, RMSE, sMAPE, MAPE (zero-handled), MASE
├── visualisation.py                            # High-resolution plotting and diagnostic routines
├── classical_forecasting.ipynb                 # Interactive, fully executed demonstration notebook
├── generate_classical_forecasting_notebook.py  # Headless notebook builder & reproducibility script
└── README.md                                   # This module guide
```

Generated results are saved automatically to:
* `results/figures/`: High-resolution figures (`.png`).
* `results/metrics/`: Full validation and test evaluation metric tables (`.csv`).
* `results/predictions/`: Out-of-sample rolling 1-step test predictions per series (`.csv`).

---

## 2. Models Implemented

1. **Seasonal Naive (Benchmark)**: $\hat{y}_t = y_{t-12}$
2. **ARIMA($p, d, q$)**: Non-seasonal autoregressive integrated moving average.
3. **SARIMA($p, d, q) \times (P, D, Q)_{12}$**: Seasonal ARIMA capturing the 12-month annual transmission cycle.
4. **Holt-Winters**: Triple exponential smoothing with level, trend, and seasonal components ($m=12$).
5. **SARIMAX (Climate Extension)**: SARIMA incorporating lagged exogenous climate variables (`tmax_lag_2`, `tmin_lag_1`, `pdsi_lag_2`).

---

## 3. Data Partitions (Strict Zero-Leakage)

* **Training Window**: January 2002 – December 2015 (168 months)
* **Validation Window**: January 2016 – December 2017 (24 months) — Parameter selection & model validation
* **Testing Window**: January 2018 – December 2019 (24 months) — Kept untouched until final evaluation

---

## 4. Representative Geographic Scope

The models are evaluated across **5 representative microregions** capturing distinct biomes, population centers, and macro-regions across Brazil, plus a **National Aggregate** series:
* `31030` — **Belo Horizonte** (Southeast, Minas Gerais, Atlantic Rainforest)
* `52010` — **Goiânia** (Centre-West, Goiás, Cerrado)
* `23016` — **Fortaleza** (Northeast, Ceará, Caatinga)
* `13007` — **Manaus** (North, Amazonas, Amazon)
* `41009` — **Maringá** (South, Paraná, Atlantic Rainforest)
* `0` — **National Aggregate** (Continental Brazil, 558 microregions summed)

---

## 5. Execution Instructions

### A. Run Unit Tests
To verify preprocessing, models, and evaluation metric logic:
```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

### B. Re-Generate & Execute Demonstration Notebook
To re-run the entire forecasting pipeline from scratch headlessly:
```bash
.venv/bin/python modelling/generate_classical_forecasting_notebook.py
.venv/bin/python data_analysis/execute_notebook.py modelling/classical_forecasting.ipynb
```

### C. Open in Jupyter
Launch Jupyter or VS Code to interactively explore [`classical_forecasting.ipynb`](file:///Users/ektamishra/Desktop/dengue%20forecasting/modelling/classical_forecasting.ipynb).
