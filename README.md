Dengue Climate Forecast

A machine learning research project investigating the relationship between climate variables and dengue incidence over time.

Objectives

* Analyze temporal patterns and seasonality in dengue cases.
* Investigate relationships between dengue incidence and climate variables.
* Engineer time-dependent features, including lagged climate variables.
* Develop and evaluate time-series forecasting models.
* Establish a reproducible data science workflow using Git, GitHub, and DVC.

Dataset

The project uses a monthly, geographically structured dengue dataset covering 2000–2019.

The dataset includes dengue case counts, population information, geographic metadata, and climate variables.

The raw and cleaned datasets are tracked using Data Version Control (DVC), rather than being committed directly to Git.

Dataset source and redistribution permissions must be documented before the data is shared publicly.

Current Progress

* [x] Initial data inspection
* [x] Data cleaning and validation
* [x] Cleaned master dataset generated (`data/dengue_cleaned_master.csv`)
* [x] DVC initialized and data tracked (`data/*.dvc`)
* [x] Exploratory data analysis (`data_analysis/eda.ipynb`)
* [x] Feature engineering (`data_analysis/feature_engineering.ipynb`, `data/dengue_features.csv`)
* [ ] Baseline forecasting models (Seasonal Naive, Auto-ARIMA, SARIMAX)
* [ ] Advanced time-series & ML models (LightGBM, XGBoost, CatBoost)
* [ ] Model evaluation and metric comparison

Project Structure

dengue-climate-forecast/
├── data/
│   ├── .gitignore
│   ├── data_2000_2019.csv.dvc
│   ├── dengue_cleaned_master.csv.dvc
│   └── dengue_features.csv.dvc
├── data_analysis/
│   ├── clean.ipynb
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── generate_eda_notebook.py
│   ├── generate_feature_engineering_notebook.py
│   └── execute_notebook.py
├── .dvc/
├── .dvcignore
├── .gitignore
├── requirements.txt
└── README.md

Tools and Technologies

* Python
* Pandas and NumPy
* Jupyter Notebook
* Git and GitHub
* DVC

The final modeling stack will be documented as the project develops.

Reproducibility

Git tracks the code, notebooks, documentation, and DVC metadata.

DVC manages dataset versions. A configured DVC remote will be required to retrieve the actual datasets on another machine.

Project Status

This project is under active development. Forecasting methods and evaluation procedures have not yet been finalized.