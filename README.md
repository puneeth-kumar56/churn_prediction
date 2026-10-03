# Customer Churn Prediction Pipeline

This project builds and evaluates a churn-prediction pipeline for customer retention analysis. It generates a realistic synthetic customer dataset, performs exploratory data analysis (EDA), engineers features, trains multiple models, compares their performance, and produces churn-risk predictions for new customer records.

## What the pipeline does

The script in `churn_prediction_pipeline.py` includes:

- Synthetic churn dataset generation with realistic patterns and missing values
- EDA summary and visual dashboard
- Feature engineering for customer behavior and risk signals
- Preprocessing pipeline for numeric and categorical data
- Model benchmarking with:
  - Logistic Regression
  - XGBoost classifier
- Evaluation metrics such as accuracy, precision, recall, F1, ROC-AUC, and PR-AUC
- Output plots for diagnostics and top churn drivers
- Production-style inference for sample customers with risk tiers and retention playbooks

## Project files

- `churn_prediction_pipeline.py` — main training, evaluation, and inference pipeline
- `churn_eda_summary.png` — generated EDA dashboard
- `churn_model_evaluation.png` — generated model evaluation plots

## Requirements

This project uses Python and the following main libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

A virtual environment is already present in the workspace:

- `ml_env/`

## Setup

From the project root, activate the bundled environment and install dependencies if needed:

```powershell
cd C:\antigravity
.\ml_env\Scripts\Activate.ps1
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

If the dependencies are already installed, you can skip the install step.

## Run the pipeline

```powershell
cd C:\antigravity
.\ml_env\Scripts\Activate.ps1
python churn_prediction_pipeline.py
```

This will:

1. Generate the synthetic dataset
2. Run EDA and save `churn_eda_summary.png`
3. Train the models
4. Evaluate results and save `churn_model_evaluation.png`
5. Run sample inference for example customers
6. Print the final churn-risk output in the terminal

## Example output

The script prints metric summaries and churn predictions such as:

- customer ID
- churn prediction (Yes/No)
- churn probability
- risk tier (`Low Risk`, `Moderate Risk`, `High Risk`)
- recommended retention action

## Notes

- The dataset is synthetic and generated for model experimentation.
- The pipeline is designed to be deterministic with fixed random seeds.
- Outputs are saved to the current working directory unless otherwise specified.

## License

This project is intended for educational and experimental use.
