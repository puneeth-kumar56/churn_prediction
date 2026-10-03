import os
import warnings
from typing import Dict, List, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/production environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Suppress minor non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="muted")


# ==============================================================================
# 1. DATA GENERATION / SYNTHETIC DATASET
# ==============================================================================
def generate_historical_churn_data(
    n_samples: int = 3500, random_seed: int = 42
) -> pd.DataFrame:
    """Generates a realistic synthetic historical dataset with realistic correlations,

    non-linear relationships, and real-world edge cases (missing values).
    """
    np.random.seed(random_seed)

    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 72, size=n_samples)
    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.25, 0.20],
    )
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_samples), 2)

    # Total charges roughly correlates with tenure * monthly charge + noise
    noise = np.random.normal(0, 25.0, size=n_samples)
    total_charges = np.round(monthly_charges * tenure_months + noise, 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
    )
    tech_support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    avg_daily_usage_mins = np.round(
        np.random.gamma(shape=3.5, scale=18.0, size=n_samples), 1
    )
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])

    # Realistic Log-Odds for Churn (Behavioral ground truth)
    log_odds = (
        -1.80
        - 0.045 * tenure_months
        + 0.024 * monthly_charges
        + 0.85 * (contract_types == "Month-to-month")
        - 0.95 * (contract_types == "Two year")
        + 0.45 * tech_support_tickets
        - 0.020 * avg_daily_usage_mins
        + 0.40 * (payment_methods == "Electronic check")
        + 0.28 * senior_citizen
    )
    churn_probability = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (np.random.rand(n_samples) < churn_probability).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure_months": tenure_months,
            "contract_type": contract_types,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "payment_method": payment_methods,
            "tech_support_tickets": tech_support_tickets.astype(float),
            "avg_daily_usage_mins": avg_daily_usage_mins,
            "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen,
            "churn": churn,
        }
    )

    # Introduce realistic missing values (e.g., brand new users with unrecorded total charges)
    df.loc[df["tenure_months"] == 1, "total_charges"] = np.nan
    mask_support = np.random.rand(n_samples) < 0.015
    df.loc[mask_support, "tech_support_tickets"] = np.nan

    return df


# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
def perform_eda(df: pd.DataFrame, output_image_path: str = "churn_eda_summary.png") -> None:
    """Executes exploratory data analysis, logs summaries, and exports a 4-panel visual dashboard."""
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nTarget Distribution (Class Balance):")
    churn_counts = df["churn"].value_counts()
    churn_ratios = df["churn"].value_counts(normalize=True) * 100
    for label, count in churn_counts.items():
        name = "Churned (1)" if label == 1 else "Retained (0)"
        print(f"  - {name}: {count:,} samples ({churn_ratios[label]:.2f}%)")

    print("\nMissing Values Audit:")
    null_summary = df.isnull().sum()
    null_cols = null_summary[null_summary > 0]
    if len(null_cols) == 0:
        print("  - No missing values detected.")
    else:
        for col, count in null_cols.items():
            print(f"  - {col}: {count} missing ({count / len(df) * 100:.2f}%)")

    # Generate 4-panel EDA visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Historical Dataset: Key Customer Churn Signals",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Target Class Balance
    sns.countplot(
        data=df,
        x="churn",
        hue="churn",
        palette=["#2b5c8f", "#d95f02"],
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Target Class Distribution", fontweight="bold")
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_xticklabels(["Retained (0)", "Churned (1)"])
    axes[0, 0].set_ylabel("Customer Count")

    # Plot 2: Correlation Heatmap of Numerical Attributes
    numeric_cols = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "tech_support_tickets",
        "avg_daily_usage_mins",
        "churn",
    ]
    corr = df[numeric_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        vmin=-1,
        vmax=1,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Feature Correlation Matrix", fontweight="bold")

    # Plot 3: Churn Rate by Contract Type
    contract_churn = (
        df.groupby("contract_type")["churn"].mean().reset_index()
    )
    sns.barplot(
        data=contract_churn,
        x="contract_type",
        y="churn",
        hue="contract_type",
        palette="Blues_r",
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Churn Rate by Contract Type", fontweight="bold")
    axes[1, 0].set_ylabel("Proportion Churned")
    axes[1, 0].set_ylim(0, 0.6)

    # Plot 4: Density Distribution of Monthly Charges by Outcome
    sns.kdeplot(
        data=df,
        x="monthly_charges",
        hue="churn",
        common_norm=False,
        fill=True,
        palette=["#2b5c8f", "#d95f02"],
        alpha=0.4,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title(
        "Monthly Charges Distribution (Retained vs Churned)", fontweight="bold"
    )
    axes[1, 1].set_xlabel("Monthly Charges ($)")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print(f"\n[Artifact Generated] EDA Dashboard saved to: {output_image_path}")


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates domain-specific behavioral interaction metrics and proxies for friction.

    This function is deterministic and stateless, making it safe across train,
    test, and production inference sets without leaking test distributions.
    """
    df_out = df.copy()

    # 1. Support Friction Intensity: Tickets filed per month of tenure
    # High frequency in early tenure indicates immediate onboarding dissatisfaction
    df_out["tickets_per_tenure_month"] = df_out["tech_support_tickets"] / (
        df_out["tenure_months"] + 1.0
    )

    # 2. Charge-to-Usage Efficiency: Cost per active usage hour
    # High cost per hour flags under-utilizing, high-risk customers
    total_monthly_usage_hrs = (df_out["avg_daily_usage_mins"] * 30.0) / 60.0
    df_out["charge_per_usage_hr"] = df_out["monthly_charges"] / (
        total_monthly_usage_hrs + 1e-3
    )

    # 3. Contract Switching Barrier (Long-term commitment vs month-to-month)
    df_out["is_long_term_contract"] = df_out["contract_type"].isin(
        ["One year", "Two year"]
    ).astype(int)

    # 4. Monetary Accumulation Ratio
    df_out["cumulative_charge_ratio"] = df_out["total_charges"] / (
        df_out["monthly_charges"] + 1e-3
    )

    return df_out


# ==============================================================================
# 4. PREPROCESSING PIPELINE & LEAK-FREE FEATURIZATION
# ==============================================================================
def build_preprocessor(
    num_features: List[str], cat_features: List[str]
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer.

    Enforces strict featurization: imputers and scalers are fitted ONLY on
    training sets and applied identically to test and unseen samples.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


# ==============================================================================
# 5. MODEL TRAINING & CROSS-VALIDATION
# ==============================================================================
def train_and_benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[Pipeline, Pipeline]:
    """Fits both a baseline model (Logistic Regression with class balancing) and

    an ensemble model (XGBoost Classifier with scale_pos_weight). Evaluates both
    via 5-Fold Stratified Cross-Validation.
    """
    print("\n" + "=" * 70)
    print("MODEL BENCHMARKING & CROSS-VALIDATION (TRAIN SET)")
    print("=" * 70)

    # 1. Baseline Model: L2-Penalized Logistic Regression
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
            ),
        ]
    )

    # 2. Advanced Ensemble: XGBoost Classifier
    # Compute inverse class ratio for positive churn weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    pos_weight = neg_count / pos_count

    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=160,
                    learning_rate=0.04,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    scale_pos_weight=pos_weight,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    # 5-Fold Stratified Cross-Validation for ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_baseline_scores = cross_val_score(
        baseline_pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
    )
    cv_xgb_scores = cross_val_score(
        xgb_pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
    )

    print(
        f"Baseline Logistic Regression  - 5-Fold Mean ROC-AUC: {cv_baseline_scores.mean():.4f} "
        f"(+/- {cv_baseline_scores.std():.4f})"
    )
    print(
        f"XGBoost Classifier Ensemble   - 5-Fold Mean ROC-AUC: {cv_xgb_scores.mean():.4f} "
        f"(+/- {cv_xgb_scores.std():.4f})"
    )

    # Fit both models on the full training partition
    baseline_pipeline.fit(X_train, y_train)
    xgb_pipeline.fit(X_train, y_train)

    return baseline_pipeline, xgb_pipeline


# ==============================================================================
# 6. EVALUATION, DIAGNOSTIC METRICS & CHARTS
# ==============================================================================
def evaluate_models(
    baseline_pipeline: Pipeline,
    xgb_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_image_path: str = "churn_model_evaluation.png",
) -> None:
    """Computes comprehensive test metrics, prints classification reports, and generates

    a 3-panel figure: Confusion Matrix, ROC-AUC Comparison, and Feature Importance.
    """
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION & MODEL COMPARISON")
    print("=" * 70)

    # Predictions
    y_pred_base = baseline_pipeline.predict(X_test)
    y_prob_base = baseline_pipeline.predict_proba(X_test)[:, 1]

    y_pred_xgb = xgb_pipeline.predict(X_test)
    y_prob_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

    # Metrics Summary
    metrics = {
        "Model": ["Logistic Regression (Baseline)", "XGBoost Classifier (Ensemble)"],
        "Accuracy": [
            accuracy_score(y_test, y_pred_base),
            accuracy_score(y_test, y_pred_xgb),
        ],
        "Precision": [
            precision_score(y_test, y_pred_base),
            precision_score(y_test, y_pred_xgb),
        ],
        "Recall": [
            recall_score(y_test, y_pred_base),
            recall_score(y_test, y_pred_xgb),
        ],
        "F1-Score": [
            f1_score(y_test, y_pred_base),
            f1_score(y_test, y_pred_xgb),
        ],
        "ROC-AUC": [
            roc_auc_score(y_test, y_prob_base),
            roc_auc_score(y_test, y_prob_xgb),
        ],
        "PR-AUC": [
            average_precision_score(y_test, y_prob_base),
            average_precision_score(y_test, y_prob_xgb),
        ],
    }
    metrics_df = pd.DataFrame(metrics).set_index("Model")
    print(metrics_df.round(4).to_string())

    print("\nXGBoost Detailed Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_xgb, target_names=["Retained", "Churned"]))

    # --------------------------------------------------------------------------
    # Visual Diagnostics Generation
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Diagnostic Performance & Feature Attribution",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Confusion Matrix (XGBoost)
    cm = confusion_matrix(y_test, y_pred_xgb)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Retained", "Churned"]
    )
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("XGBoost Confusion Matrix", fontweight="bold")
    axes[0].grid(False)

    # 2. ROC Curves Comparison
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    auc_base = roc_auc_score(y_test, y_prob_base)
    auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    axes[1].plot(
        fpr_base,
        tpr_base,
        linestyle="--",
        label=f"Baseline LogReg (AUC = {auc_base:.3f})",
        color="#7f7f7f",
    )
    axes[1].plot(
        fpr_xgb,
        tpr_xgb,
        linewidth=2.2,
        label=f"Champion XGBoost (AUC = {auc_xgb:.3f})",
        color="#1f77b4",
    )
    axes[1].plot([0, 1], [0, 1], "k:", alpha=0.6)
    axes[1].set_title("ROC Curve Benchmark", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    # 3. XGBoost Feature Importance (Gain)
    preprocessor = xgb_pipeline.named_steps["preprocessor"]
    encoded_feature_names = preprocessor.get_feature_names_out()
    cleaned_feature_names = [f.split("__")[-1] for f in encoded_feature_names]

    xgb_model = xgb_pipeline.named_steps["classifier"]
    importances = xgb_model.feature_importances_

    importance_df = pd.DataFrame(
        {"Feature": cleaned_feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    top_features = importance_df.head(8)
    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=axes[2],
    )
    axes[2].set_title("Top Churn Drivers (XGBoost Gain)", fontweight="bold")
    axes[2].set_xlabel("Relative Importance")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print(f"[Artifact Generated] Evaluation Plots saved to: {output_image_path}")


# ==============================================================================
# 7. PRODUCTION INFERENCE ENGINE & RETENTION DECISIONING
# ==============================================================================
def predict_trend(
    new_data: Union[Dict, pd.DataFrame],
    model_pipeline: Pipeline,
    feature_columns: List[str],
    classification_threshold: float = 0.50,
) -> pd.DataFrame:
    """Processes unseen raw customer records, applies feature transformations, runs

    model inference, and assigns risk tiers with tailored retention playbooks.
    """
    if isinstance(new_data, dict):
        df_input = pd.DataFrame([new_data])
    else:
        df_input = new_data.copy()

    # Apply stateless feature engineering
    df_engineered = engineer_features(df_input)

    # Verify and select expected features
    missing_cols = set(feature_columns) - set(df_engineered.columns)
    if missing_cols:
        raise ValueError(f"Input records are missing required columns: {missing_cols}")

    X_infer = df_engineered[feature_columns]

    # Model Inference
    probabilities = model_pipeline.predict_proba(X_infer)[:, 1]
    predictions = (probabilities >= classification_threshold).astype(int)

    results = []
    customer_ids = df_input.get(
        "customer_id", [f"UNKNOWN-{i}" for i in range(len(df_input))]
    )

    for cid, prob, pred in zip(customer_ids, probabilities, predictions):
        # Business Risk Tier Stratification
        if prob >= 0.70:
            risk_tier = "High Risk"
            recommendation = (
                "Priority 1: Dedicated CS Outreach + 25% 1-year contract extension incentive."
            )
        elif prob >= 0.40:
            risk_tier = "Moderate Risk"
            recommendation = (
                "Priority 2: Product engagement check-in + tech support satisfaction audit."
            )
        else:
            risk_tier = "Low Risk"
            recommendation = (
                "Priority 3: Standard nurture cycle + feature adoption newsletter."
            )

        results.append(
            {
                "customer_id": cid,
                "churn_predicted": "Yes" if pred == 1 else "No",
                "churn_probability": round(float(prob), 4),
                "risk_tier": risk_tier,
                "actionable_retention_playbook": recommendation,
            }
        )

    return pd.DataFrame(results)


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================
def main():
    print("=" * 70)
    print("INITIALIZING CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Ingest Data
    raw_df = generate_historical_churn_data(n_samples=4000, random_seed=42)

    # 2. EDA Phase
    perform_eda(raw_df, output_image_path="churn_eda_summary.png")

    # 3. Apply Feature Engineering
    engineered_df = engineer_features(raw_df)

    # 4. Partition Features and Target
    target = "churn"
    drop_columns = ["customer_id", target]
    feature_cols = [c for c in engineered_df.columns if c not in drop_columns]

    X = engineered_df[feature_cols]
    y = engineered_df[target]

    # Explicit column types for the ColumnTransformer
    num_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "tech_support_tickets",
        "avg_daily_usage_mins",
        "tickets_per_tenure_month",
        "charge_per_usage_hr",
        "cumulative_charge_ratio",
    ]
    cat_features = [
        "contract_type",
        "payment_method",
        "paperless_billing",
        "senior_citizen",
        "is_long_term_contract",
    ]

    # 5. Stratified Train-Test Split (Strict Isolation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\nTrain Set: {X_train.shape[0]} records | Test Set: {X_test.shape[0]} records")

    # 6. Build Preprocessor and Train Models
    preprocessor = build_preprocessor(num_features, cat_features)
    baseline_model, champion_xgb = train_and_benchmark_models(
        X_train, y_train, preprocessor
    )

    # 7. Model Evaluation
    evaluate_models(
        baseline_model,
        champion_xgb,
        X_test,
        y_test,
        output_image_path="churn_model_evaluation.png",
    )

    # 8. Test Sample Production Inference Pipeline
    print("\n" + "=" * 70)
    print("TESTING PRODUCTION INFERENCE ENGINE (SAMPLE CUSTOMERS)")
    print("=" * 70)

    unseen_customers = pd.DataFrame(
        [
            {
                "customer_id": "CUST-90001",
                "tenure_months": 2,
                "contract_type": "Month-to-month",
                "monthly_charges": 105.50,
                "total_charges": 210.00,
                "payment_method": "Electronic check",
                "tech_support_tickets": 4.0,
                "avg_daily_usage_mins": 18.0,
                "paperless_billing": "Yes",
                "senior_citizen": 0,
            },
            {
                "customer_id": "CUST-90002",
                "tenure_months": 36,
                "contract_type": "Two year",
                "monthly_charges": 42.00,
                "total_charges": 1512.00,
                "payment_method": "Credit card",
                "tech_support_tickets": 0.0,
                "avg_daily_usage_mins": 90.0,
                "paperless_billing": "No",
                "senior_citizen": 0,
            },
            {
                "customer_id": "CUST-90003",
                "tenure_months": 8,
                "contract_type": "Month-to-month",
                "monthly_charges": 75.00,
                "total_charges": 600.00,
                "payment_method": "Bank transfer",
                "tech_support_tickets": 2.0,
                "avg_daily_usage_mins": 45.0,
                "paperless_billing": "Yes",
                "senior_citizen": 1,
            },
        ]
    )

    inference_output = predict_trend(
        new_data=unseen_customers,
        model_pipeline=champion_xgb,
        feature_columns=feature_cols,
        classification_threshold=0.50,
    )
    print(inference_output.to_string(index=False))
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()