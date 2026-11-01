# Brazilian Dengue Climate Forecasting: Setup & Reproduction Guide

This guide provides instructions for setting up, executing, and reproducing the **Brazilian Dengue Climate Forecasting & Analytics Project** on macOS, Linux, and Windows environments.

---

## 📋 Table of Contents
1. [System & Hardware Requirements](#1-system--hardware-requirements)
2. [Python Version Requirements](#2-python-version-requirements)
3. [Step-by-Step Installation](#3-step-by-step-installation)
   - [3.1 Clone the Repository](#31-clone-the-repository)
   - [3.2 Virtual Environment Creation & Activation](#32-virtual-environment-creation--activation)
   - [3.3 Package Installation](#33-package-installation)
   - [3.4 Jupyter Kernel Registration](#34-jupyter-kernel-registration)
4. [Dataset Retrieval via DVC](#4-dataset-retrieval-via-dvc)
   - [4.1 How DVC Manages Data](#41-how-dvc-manages-data)
   - [4.2 Configuring a DVC Remote](#42-configuring-a-dvc-remote)
   - [4.3 Pulling Data & Integrity Verification](#43-pulling-data--integrity-verification)
   - [4.4 Alternative Manual Placement](#44-alternative-manual-placement)
5. [Launching JupyterLab & Running Research Notebooks](#5-launching-jupyterlab--running-research-notebooks)
   - [5.1 Interactive Execution](#51-interactive-execution)
   - [5.2 Headless / CLI Execution](#52-headless--cli-execution)
6. [Running the Automated Test Suite](#6-running-the-automated-test-suite)
7. [Launching the Interactive Streamlit Dashboard (DengueCast)](#7-launching-the-interactive-streamlit-dashboard-denguecast)
8. [Expected Repository Structure & Output Directories](#8-expected-repository-structure--output-directories)
9. [Troubleshooting & Common Questions](#9-troubleshooting--common-questions)

---

## 1. System & Hardware Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | macOS 12+, Ubuntu 20.04+, Windows 10/11 | macOS 14+ (Apple Silicon/Intel), Ubuntu 22.04+, Windows 11 (WSL2) |
| **Processor** | Dual-core 2.0 GHz CPU | 4+ physical cores (Apple M-series or Intel/AMD x86_64) |
| **Memory (RAM)** | 8 GB | 16 GB (required for fast panel feature engineering across 558 microregions) |
| **Disk Space** | 2 GB free space | 5 GB free space (to store raw, cleaned, and feature-engineered CSV files) |

---

## 2. Python Version Requirements

- **Supported Versions**: Python **3.10**, **3.11**, and **3.12**.
- **Tested Reference**: Python **3.11.16**.

> [!IMPORTANT]
> Verify your Python version before creating a virtual environment:
> ```bash
> python3 --version   # macOS / Linux
> python --version    # Windows
> ```
> If your system default is Python 3.9 or older, please install Python 3.11 from [python.org](https://www.python.org/downloads/) or via your system package manager (e.g. `brew install python@3.11` on macOS, `sudo apt install python3.11 python3.11-venv` on Ubuntu).

---

## 3. Step-by-Step Installation

### 3.1 Clone the Repository

Clone the project from GitHub and navigate to the project root:

```bash
git clone https://github.com/<your-username-or-org>/dengue-climate-forecasting.git
cd dengue-climate-forecasting
```

*(If cloning into a folder with spaces like `dengue forecasting`, remember to wrap path arguments in quotes).*

---

### 3.2 Virtual Environment Creation & Activation

Always isolate the project dependencies within a local virtual environment named `.venv`.

#### On macOS & Linux
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

#### On Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv .venv

# Enable script execution if restricted (run once per user session)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt)
```cmd
:: Create virtual environment
python -m venv .venv

:: Activate virtual environment
.venv\Scripts\activate.bat
```

When activated, your terminal prompt will display `(.venv)`.

---

### 3.3 Package Installation

Upgrade standard packaging tools and install all dependencies from [`requirements.txt`](file:///Users/ektamishra/Desktop/dengue%20forecasting/requirements.txt):

```bash
# Upgrade pip and packaging utilities
pip install --upgrade pip setuptools wheel

# Install all project dependencies
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` contains the full dependency tree, including:
> - **Core Data Science & Modeling**: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`
> - **Data Versioning**: `dvc`, `dvc-http`
> - **Visualisation**: `matplotlib`, `seaborn`, `plotly`
> - **Interactive Dashboard**: `streamlit`
> - **Notebooks & Kernel**: `jupyterlab`, `ipykernel`, `nbformat`, `nbclient`
> - **Testing**: `pytest`
> Platform-specific wheels (e.g. `appnope` on macOS) are protected with standard environment markers to ensure seamless installation on Windows and Linux.

---

### 3.4 Jupyter Kernel Registration

To allow JupyterLab to recognize your isolated virtual environment as a selectable execution kernel:

```bash
python -m ipykernel install --user --name dengue-env --display-name "Python (Dengue Env)"
```

---

## 4. Dataset Retrieval via DVC

### 4.1 How DVC Manages Data

To prevent Git repository bloat, large data files are tracked using [Data Version Control (DVC)](https://dvc.org/). The Git repository contains only lightweight pointer files:

- `data/data_2000_2019.csv.dvc` (Raw dataset, 31.8 MB)
- `data/dengue_cleaned_master.csv.dvc` (Cleaned dataset, 31.6 MB)
- `data/dengue_features.csv.dvc` (Engineered features dataset, 159.2 MB)

The actual `.csv` files are listed in `data/.gitignore` and must be retrieved using DVC or placed in the `data/` directory.

---

### 4.2 Configuring a DVC Remote

If your institution, team, or lab hosts the dataset on a remote storage service, configure DVC with the appropriate URL:

#### Amazon S3
```bash
dvc remote add -d myremote s3://<your-s3-bucket-name>/dvcstore
# Authenticate using standard AWS CLI credentials:
# aws configure
```

#### Google Cloud Storage (GCS)
```bash
dvc remote add -d myremote gs://<your-gcs-bucket-name>/dvcstore
# Authenticate using Google Cloud SDK:
# gcloud auth application-default login
```

#### Google Drive
```bash
dvc remote add -d myremote gdrive://<your-google-drive-folder-id>
# On first pull, DVC will prompt for a one-time OAuth browser login.
```

#### Shared Local / Network Storage (e.g. NFS, external drive)
```bash
# macOS / Linux
dvc remote add -d myremote /Volumes/SharedStorage/dengue_dvc_cache

# Windows
dvc remote add -d myremote D:\SharedStorage\dengue_dvc_cache
```

> [!WARNING]
> Never commit private tokens, passwords, or cloud credentials to Git. Use environment variables (e.g., `AWS_ACCESS_KEY_ID`, `GOOGLE_APPLICATION_CREDENTIALS`) or default provider keychains.

---

### 4.3 Pulling Data & Integrity Verification

Once a remote is configured, retrieve the tracked datasets:

```bash
# Download datasets from configured remote
dvc pull

# Verify that all tracked datasets are present and match their MD5 hashes
dvc status
```

Expected `dvc status` output when all data is present:
```
Data and pipelines are up to date.
```

---

### 4.4 Alternative Manual Placement

If you received the datasets directly (e.g., via Zenodo, institutional download, or shared archive), copy the following files directly into the [`data/`](file:///Users/ektamishra/Desktop/dengue%20forecasting/data) directory:

1. `data/data_2000_2019.csv`
2. `data/dengue_cleaned_master.csv`
3. `data/dengue_features.csv`

Verify that their hashes match the metadata:
```bash
dvc status
```

---

## 5. Launching JupyterLab & Running Research Notebooks

### 5.1 Interactive Execution

Start JupyterLab in your browser:

```bash
jupyter lab
```

When opening any notebook, ensure the kernel in the top-right corner is set to **`Python (Dengue Env)`**.

Execute notebooks in the following chronological research sequence:

```
Step 1: data_analysis/clean.ipynb
└── Validates raw data, imputes missing values, links microregions,
    and produces data/dengue_cleaned_master.csv.

Step 2: data_analysis/eda.ipynb
└── Performs exploratory data analysis, seasonal decomposition,
    epidemic wave analysis, zero-inflation and log1p skewness audits.

Step 3: data_analysis/feature_engineering.ipynb
└── Generates 1–6 month lags, 3/6/12-month rolling statistics, environmental
    interactions, and cyclic features, producing data/dengue_features.csv.

Step 4: modelling/classical_forecasting.ipynb
└── Conducts stationarity tests (ADF/KPSS), differencing variance audits,
    fits Seasonal Naive, ARIMA, SARIMA, Holt-Winters, and SARIMAX,
    and generates evaluation metrics and out-of-sample predictions.
```

---

### 5.2 Headless / CLI Execution

To execute notebooks headlessly from the command line without opening a browser:

```bash
# On macOS / Linux:
PYTHONPATH=. python data_analysis/execute_notebook.py data_analysis/clean.ipynb
PYTHONPATH=. python data_analysis/execute_notebook.py data_analysis/eda.ipynb
PYTHONPATH=. python data_analysis/execute_notebook.py data_analysis/feature_engineering.ipynb
PYTHONPATH=. python data_analysis/execute_notebook.py modelling/classical_forecasting.ipynb

# On Windows (PowerShell):
$env:PYTHONPATH="."
python data_analysis\execute_notebook.py data_analysis\clean.ipynb
python data_analysis\execute_notebook.py data_analysis\eda.ipynb
python data_analysis\execute_notebook.py data_analysis\feature_engineering.ipynb
python data_analysis\execute_notebook.py modelling\classical_forecasting.ipynb
```

---

## 6. Running the Automated Test Suite

The project includes unit and integration tests covering preprocessing, stationarity, model estimation, evaluation metrics, and dashboard data loaders:

### Run All Tests
```bash
# macOS / Linux
PYTHONPATH=. pytest tests/

# Windows (PowerShell)
$env:PYTHONPATH="."
pytest tests/

# Windows (Command Prompt)
set PYTHONPATH=.
pytest tests/
```

### Run with Verbose Output
```bash
PYTHONPATH=. pytest -v tests/
```

### Test Scope
- [`tests/test_preprocessing.py`](file:///Users/ektamishra/Desktop/dengue%20forecasting/tests/test_preprocessing.py): Validates temporal train/val/test splits (2002–2015, 2016–2017, 2018–2019), absence of data leakage, and panel extraction.
- [`tests/test_models.py`](file:///Users/ektamishra/Desktop/dengue%20forecasting/tests/test_models.py): Tests model instantiation and 1-step rolling out-of-sample forecast generation.
- [`tests/test_evaluation.py`](file:///Users/ektamishra/Desktop/dengue%20forecasting/tests/test_evaluation.py): Tests MAE, RMSE, sMAPE, MASE, and zero-safe MAPE handling on zero/near-zero dengue incidence.
- [`tests/test_dashboard.py`](file:///Users/ektamishra/Desktop/dengue%20forecasting/tests/test_dashboard.py): Tests data loader caching, metric consistency, prediction schemas, and runs an end-to-end simulated `AppTest` across all 6 Streamlit sections.

Expected Result:
```
======================== 16 passed, 1 warning in ~7s =========================
```

---

## 7. Launching the Interactive Streamlit Dashboard (DengueCast)

**DengueCast** (*Brazilian Dengue Climate Forecasting & Analytics Platform*) is a demonstration dashboard built with Streamlit and Plotly.

### Launch Command

Ensure your virtual environment is active, then run:

```bash
# macOS / Linux
streamlit run dashboard/app.py

# Windows
streamlit run dashboard\app.py
```

The application will launch locally at **`http://localhost:8501`**.

### Dashboard Features

1. **Executive Overview**: High-level KPIs across 558 microregions, 20-year coverage, model rankings, and champion summary.
2. **Exploratory Data Analysis (EDA)**: Monthly incidence curves, seasonal peaks (March–April), zero-inflation audits, and biome comparisons.
3. **Climate & Environmental Drivers**: Temperature heatmaps, PDSI anomalies, lagged cross-correlations (1–3 months lead), and scientific notes on correlation vs. causation.
4. **Stationarity & Diagnostics**: Interactive ADF and KPSS hypothesis testing, differencing variance reductions, and rolling statistics.
5. **Classical Model Comparison**: Side-by-side benchmark comparison (Seasonal Naive, ARIMA, SARIMA, Holt-Winters, SARIMAX) across $\log(1+x)$ and original incidence scales.
6. **Interactive Forecast Explorer**: Visualise actual vs. predicted rolling 1-step forecasts and residual errors across 6 biomes/benchmarks and validation/test splits.

---

## 8. Expected Repository Structure & Output Directories

```
dengue-climate-forecasting/
├── .dvc/                             # DVC metadata configuration
├── .dvcignore                        # DVC ignore specifications
├── .gitignore                        # Git ignore patterns (ignores data CSVs, virtualenvs)
├── README.md                         # Main research overview
├── SETUP.md                          # Comprehensive setup & reproduction guide (this document)
├── requirements.txt                  # Full pinned dependencies for cross-platform replication
│
├── data/                             # Dataset directory (CSVs tracked via DVC)
│   ├── .gitignore                    # Ensures data CSVs are not committed to Git
│   ├── data_2000_2019.csv.dvc        # Pointer to raw dataset (31.8 MB)
│   ├── dengue_cleaned_master.csv.dvc # Pointer to cleaned dataset (31.6 MB)
│   └── dengue_features.csv.dvc       # Pointer to engineered features dataset (159.2 MB)
│
├── data_analysis/                    # Data exploration & feature engineering pipelines
│   ├── clean.ipynb                   # Data cleaning & imputation notebook
│   ├── eda.ipynb                     # Exploratory data analysis notebook
│   ├── feature_engineering.ipynb     # Feature construction & lag generation notebook
│   ├── generate_eda_notebook.py      # Automated EDA notebook generator
│   ├── generate_feature_engineering_notebook.py
│   └── execute_notebook.py           # Headless notebook execution runner
│
├── modelling/                        # Classical time-series modeling module
│   ├── config.py                     # Split dates, hyperparameter grids, locations
│   ├── preprocessing.py              # Time series indexing, split partitioning, scaling
│   ├── stationarity.py               # ADF & KPSS tests, differencing variance audits
│   ├── models.py                     # Seasonal Naive, ARIMA, SARIMA, Holt-Winters, SARIMAX
│   ├── evaluation.py                 # MAE, RMSE, sMAPE, MASE, zero-safe MAPE metrics
│   ├── visualisation.py              # Publication-quality matplotlib plotting functions
│   ├── classical_forecasting.ipynb   # Comprehensive modeling and evaluation notebook
│   └── README.md                     # Technical modeling methodology documentation
│
├── dashboard/                        # Interactive Streamlit demonstration dashboard
│   ├── app.py                        # Main Streamlit application
│   ├── data_loader.py                # Cached data access layer
│   ├── requirements.txt              # Focused dashboard requirements
│   └── README.md                     # Dashboard navigation guide
│
├── results/                          # Generated research outputs
│   ├── figures/                      # Publication figures (stationarity, ACF, forecasts)
│   ├── metrics/                      # Model comparison tables (validation & test CSVs)
│   └── predictions/                  # Out-of-sample predictions (val & test CSVs)
│
├── reports/                          # Formal research evaluation reports
│   └── phase2_midsemester_report.md  # Consolidated Phase 2 evaluation report
│
└── tests/                            # Automated test suite
    ├── test_preprocessing.py         # Split & leakage tests
    ├── test_models.py                # Time series model tests
    ├── test_evaluation.py            # Evaluation metric correctness tests
    └── test_dashboard.py             # Dashboard data integrity & AppTest tests
```

---

## 9. Troubleshooting & Common Questions

### Q1: `ModuleNotFoundError: No module named 'modelling'` or `'dashboard'`
Ensure that `PYTHONPATH` includes the repository root directory when running scripts or tests:
- **macOS / Linux**: `export PYTHONPATH=.` or run as `PYTHONPATH=. pytest tests/`
- **Windows (PowerShell)**: `$env:PYTHONPATH="."`
- **Windows (CMD)**: `set PYTHONPATH=.`

### Q2: Windows PowerShell script execution error
If you see `File ... cannot be loaded because running scripts is disabled on this system`, run PowerShell as Administrator or user and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q3: `dvc pull` fails with `No remote found`
By default, the repository comes with `.dvc` tracking files, but no default remote URL is hardcoded to avoid exposing private endpoints. Follow [Section 4.2](#42-configuring-a-dvc-remote) to configure your storage remote or manually copy the files into `data/`.

### Q4: Streamlit port already in use
If port `8501` is already occupied by another service:
```bash
streamlit run dashboard/app.py --server.port 8502
```

### Q5: Out of Memory during Feature Engineering
`data_analysis/feature_engineering.ipynb` processes a 133,920-row $\times$ 95-column panel. If running on a machine with less than 8 GB of RAM:
1. Close memory-intensive background applications.
2. In `data_analysis/feature_engineering.ipynb`, downcast float64 columns to float32 before computing rolling windows.
