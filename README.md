# customer-churn-analysis

Short, reproducible exploratory analysis of customer churn using Jupyter Notebook(s).
# [Live app](https://customer2619.streamlit.app)

## Table of contents
- [Project overview](#project-overview)
- [Repository contents](#repository-contents)
- [Notebook: CCA.ipynb (summary)](#notebook-ccaipynb-summary)
- [Data](#data)
- [Installation & dependencies](#installation--dependencies)
- [How to run](#how-to-run)
- [Reproducibility & outputs](#reproducibility--outputs)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project overview
This repository contains an exploratory customer churn analysis performed in a single Jupyter notebook. The analysis focuses on cleaning and exploring the provided telecom-style churn dataset to surface patterns and drivers of churn (no production model training is included in the notebook).

## Repository contents
- CCA.ipynb — Main Jupyter notebook with data loading, cleaning, exploratory data analysis, visualizations, and short interpretations.
- Customer Churn.csv — Original dataset used by the notebook.
- Customer Churn Analysis report.pdf — A PDF report that summarizes the work (visuals and findings exported from the notebook).

## Notebook: CCA.ipynb (summary)
Purpose: perform EDA and surface feature relationships with churn.

Key steps
- Load dataset: Customer Churn.csv
- Data cleaning:
  - Replaced blank TotalCharges entries with 0 and converted TotalCharges to numeric.
  - Converted SeniorCitizen from 0/1 to "yes"/"no" for readability.
- Exploratory analysis and visualizations:
  - Churn class distribution (counts and percentage).
  - Demographic and service-feature breakdowns (gender, SeniorCitizen, tenure, Contract, PaymentMethod).
  - Service adoption plots for PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
  - Payment method analysis.

Key findings (from the notebook)
- Overall churn rate: approximately 26.5% of customers have churned.
- Tenure: customers with very short tenure (e.g., 1–2 months) show substantially higher churn; longer-tenured customers tend to stay.
- Contract: month-to-month customers are far more likely to churn than customers on 1- or 2-year contracts.
- Senior citizens show a higher comparative churn percentage.
- Payment method: customers using Electronic check show a higher proportion of churn relative to other payment methods.
- Service relationships: customers without some protective services (e.g., OnlineSecurity, TechSupport) or on certain Internet service types show differing churn patterns (visual summaries in notebook).

Notes:
- The notebook is EDA-focused; it does not include model training or hyperparameter tuning.
- For full visuals and step-by-step code, open CCA.ipynb.

## Data
- Place local or derived datasets in a data/ folder if you expand the project (recommended layout):
  - data/raw/ — original CSV(s)
  - data/processed/ — cleaned / feature-engineered outputs
- Current dataset in repo: Customer Churn.csv (check for any privacy/usage restrictions before sharing).

## Installation & dependencies
Recommended: create an isolated Python environment.

Conda
- conda create -n churn python=3.10 -y
- conda activate churn
- pip install -r requirements.txt

Pip / venv
- python -m venv .venv
- source .venv/bin/activate  # Windows: .venv\Scripts\activate
- pip install -r requirements.txt

Minimal recommended packages (add to requirements.txt)
- pandas
- numpy
- matplotlib
- seaborn
- jupyterlab or notebook
- nbconvert
- scikit-learn

Optional (if you expand to modeling/interpretability)
- xgboost
- shap
- joblib

Install kernel for notebook:
- python -m ipykernel install --user --name=churn-env --display-name "Churn Env"

## How to run
1. Activate your environment and start Jupyter:
   - jupyter lab
   - or jupyter notebook
2. Open CCA.ipynb and run cells (Kernel → Restart & Run All) to reproduce the analysis.
3. To execute notebooks headlessly and save executed versions:
   - pip install nbconvert nbclient
   - jupyter nbconvert --to notebook --execute CCA.ipynb --output executed-CCA.ipynb

## Reproducibility & outputs
- Save cleaned/processed datasets and figures to an artifacts/ or outputs/ directory.
- Use consistent random seeds when adding modeling steps (numpy, scikit-learn).
- Persist models or intermediary objects with joblib.dump or pickle (if added later).
- If you automate notebook execution in CI, re-run notebooks and commit executed outputs or export key results to a reports/ folder.

## Contributing
- Open issues for bugs or feature requests.
- For changes to notebooks, re-run the notebook so outputs are consistent before submitting a PR.
- Add new Python package dependencies to requirements.txt and document any data-sourcing steps.


## Contact
Repository owner: daya-2619
