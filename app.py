
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, accuracy_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

import shap

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease Screening App",
    page_icon="❤️",
    layout="wide"
)

DATA_PATH = Path("heart_disease_health_indicators.csv")
TARGET = "HeartDiseaseorAttack"
RANDOM_STATE = 42

FEATURE_LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Had cholesterol check",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoker",
    "Stroke": "History of stroke",
    "Diabetes": "Diabetes",
    "PhysActivity": "Physical activity",
    "Fruits": "Consumes fruits",
    "Veggies": "Consumes vegetables",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has healthcare coverage",
    "NoDocbcCost": "Could not see doctor due to cost",
    "GenHlth": "General health (1=excellent, 5=poor)",
    "MentHlth": "Poor mental health days (0–30)",
    "PhysHlth": "Poor physical health days (0–30)",
    "DiffWalk": "Difficulty walking",
    "Sex": "Sex (0=female, 1=male)",
    "Age": "Age category (1–13)",
    "Education": "Education level (1–6)",
    "Income": "Income level (1–8)"
}

BINARY_FIELDS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex"
]
ORDINAL_FIELDS = {
    "Diabetes": (0, 2),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}
CONTINUOUS_FIELDS = {
    "BMI": (10, 80)
}

# ============================================================
# HELPERS
# ============================================================
def optimize_threshold(y_true, y_prob, min_precision=0.25):
    threshold_grid = np.arange(0.01, 1.00, 0.01)
    rows = []

    for t in threshold_grid:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "accuracy": accuracy_score(y_true, y_pred_t),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_t),
            "precision": precision_score(y_true, y_pred_t, zero_division=0),
            "recall": recall_score(y_true, y_pred_t, zero_division=0),
            "f1": f1_score(y_true, y_pred_t, zero_division=0)
        })

    df_thr = pd.DataFrame(rows)
    best_f1 = df_thr.loc[df_thr["f1"].idxmax()].copy()
    constrained = df_thr[df_thr["precision"] >= min_precision].copy()
    best_recall = None
    if len(constrained) > 0:
        best_recall = constrained.sort_values(
            ["recall", "f1", "balanced_accuracy"], ascending=False
        ).iloc[0].copy()
    return df_thr, best_f1, best_recall

@st.cache_data
def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    raise FileNotFoundError(
        "Dataset not found. Upload heart_disease_health_indicators.csv or place it beside app.py."
    )

@st.cache_resource
def train_model(df):
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(int).copy()

    numeric_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features)
        ],
        remainder="drop"
    )

    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Please install from requirements.txt")

    model = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.50).astype(int)

    thresholds_df, best_f1, best_recall = optimize_threshold(
        y_true=y_test, y_prob=y_prob, min_precision=0.25
    )
    selected_threshold = float(best_recall["threshold"]) if best_recall is not None else float(best_f1["threshold"])
    y_pred_selected = (y_prob >= selected_threshold).astype(int)

    metrics_default = {
        "accuracy": accuracy_score(y_test, y_pred_default),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_default),
        "precision": precision_score(y_test, y_pred_default, zero_division=0),
        "recall": recall_score(y_test, y_pred_default, zero_division=0),
        "f1": f1_score(y_test, y_pred_default, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    metrics_selected = {
        "accuracy": accuracy_score(y_test, y_pred_selected),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_selected),
        "precision": precision_score(y_test, y_pred_selected, zero_division=0),
        "recall": recall_score(y_test, y_pred_selected, zero_division=0),
        "f1": f1_score(y_test, y_pred_selected, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_model = model.named_steps["model"]

    # SHAP on transformed test sample
    X_test_transformed = fitted_preprocessor.transform(X_test)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=numeric_features, index=X_test.index)

    sample_n = min(2000, len(X_test_transformed_df))
    X_shap = X_test_transformed_df.sample(sample_n, random_state=RANDOM_STATE)

    try:
        explainer = shap.TreeExplainer(fitted_model)
        shap_values = explainer.shap_values(X_shap)
    except Exception:
        explainer = shap.Explainer(fitted_model, X_shap)
        shap_values = explainer(X_shap)
        shap_values = shap_values.values if hasattr(shap_values, "values") else shap_values

    if isinstance(shap_values, list):
        shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_array = shap_values

    if len(np.shape(shap_array)) == 3:
        shap_array = shap_array[:, :, 1]

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return {
        "df": df,
        "X_columns": numeric_features,
        "model": model,
        "preprocessor": fitted_preprocessor,
        "xgb_model": fitted_model,
        "X_test": X_test,
        "y_test": y_test,
        "thresholds_df": thresholds_df,
        "best_f1": best_f1,
        "best_recall": best_recall,
        "selected_threshold": selected_threshold,
        "metrics_default": metrics_default,
        "metrics_selected": metrics_selected,
        "X_shap": X_shap,
        "shap_array": shap_array,
        "shap_importance_df": shap_importance_df,
    }

def make_prediction(user_input_df, assets, threshold):
    prob = assets["model"].predict_proba(user_input_df)[:, 1][0]
    pred = int(prob >= threshold)
    return prob, pred

def plot_threshold_tradeoff(thresholds_df):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(thresholds_df["threshold"], thresholds_df["precision"], label="Precision")
    ax.plot(thresholds_df["threshold"], thresholds_df["recall"], label="Recall")
    ax.plot(thresholds_df["threshold"], thresholds_df["f1"], label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold trade-off")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_shap_bar(assets, top_n=12):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    top = assets["shap_importance_df"].head(top_n).sort_values("mean_abs_shap", ascending=True)
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_title(f"Top {top_n} SHAP drivers")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    return fig

def plot_shap_summary(assets):
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(assets["shap_array"], assets["X_shap"], show=False)
    plt.tight_layout()
    return fig

def plot_local_shap(user_input_df, assets):
    transformed = assets["preprocessor"].transform(user_input_df)
    transformed_df = pd.DataFrame(transformed, columns=assets["X_columns"])
    try:
        explainer = shap.TreeExplainer(assets["xgb_model"])
        shap_vals = explainer.shap_values(transformed_df)
    except Exception:
        explainer = shap.Explainer(assets["xgb_model"], transformed_df)
        shap_vals = explainer(transformed_df)
        shap_vals = shap_vals.values if hasattr(shap_vals, "values") else shap_vals

    if isinstance(shap_vals, list):
        local_vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        local_vals = shap_vals[0]
        if len(np.shape(local_vals)) == 2:
            local_vals = local_vals[:, 1]

    contrib = pd.DataFrame({
        "feature": transformed_df.columns,
        "shap_value": local_vals
    }).sort_values("shap_value", key=lambda s: np.abs(s), ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in contrib["shap_value"]]
    ax.barh(contrib["feature"][::-1], contrib["shap_value"][::-1], color=colors[::-1])
    ax.set_title("Local explanation: strongest contributors")
    ax.set_xlabel("SHAP contribution")
    fig.tight_layout()
    return fig, contrib

def risk_band(prob, threshold):
    if prob < threshold:
        return "Low"
    if prob < min(1.0, threshold + 0.15):
        return "Moderate"
    if prob < min(1.0, threshold + 0.35):
        return "High"
    return "Very High"

# ============================================================
# APP
# ============================================================
st.title("❤️ Heart Disease Screening App")
st.caption("Created by Powell Andile Ndlovu")
st.caption("BRFSS-based screening model with threshold optimization and SHAP interpretation.")

st.info(
    "Model used: XGBoost Classifier. "
    "This gradient-boosted tree algorithm was selected because it handles non-linear patterns well, "
    "works strongly on structured tabular data, and supports SHAP-based explanations."

    "Created by Powell Ndlovu"
)

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("Use the BRFSS-style file with the target column `HeartDiseaseorAttack` for training.")
    st.header("Threshold")
    precision_floor = st.caption("The app defaults to the trained screening threshold selected from the test set.")

try:
    df = load_dataset(uploaded_file)
    assets = train_model(df)
except Exception as e:
    st.error(f"Failed to load/train app: {e}")
    st.stop()

selected_threshold = assets["selected_threshold"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Screening panel", "Model performance", "SHAP insights", "Dataset audit"
])

with tab1:
    st.subheader("Patient screening panel")
    st.write("Enter the screening profile below. The model outputs a risk probability and a threshold-based screening decision.")

    cols = st.columns(3)
    user_values = {}

    # Binary fields
    for i, field in enumerate(BINARY_FIELDS):
        with cols[i % 3]:
            user_values[field] = st.selectbox(
                FEATURE_LABELS.get(field, field),
                options=[0, 1],
                index=0,
                key=f"bin_{field}"
            )

    # Continuous / ordinal
    for i, (field, rng) in enumerate(CONTINUOUS_FIELDS.items()):
        with cols[i % 3]:
            user_values[field] = st.slider(
                FEATURE_LABELS.get(field, field),
                min_value=int(rng[0]), max_value=int(rng[1]), value=25,
                key=f"cont_{field}"
            )

    ord_items = list(ORDINAL_FIELDS.items())
    for i, (field, rng) in enumerate(ord_items):
        default = rng[0]
        if field == "GenHlth":
            default = 3
        elif field in ["MentHlth", "PhysHlth"]:
            default = 0
        elif field == "Age":
            default = 7
        elif field == "Education":
            default = 4
        elif field == "Income":
            default = 5
        elif field == "Diabetes":
            default = 0

        with cols[(i + 1) % 3]:
            user_values[field] = st.slider(
                FEATURE_LABELS.get(field, field),
                min_value=int(rng[0]), max_value=int(rng[1]), value=int(default),
                key=f"ord_{field}"
            )

    user_input_df = pd.DataFrame([user_values])[assets["X_columns"]]

    c1, c2 = st.columns([1, 1])
    with c1:
        threshold = st.slider(
            "Operational threshold",
            min_value=0.05, max_value=0.50,
            value=float(round(selected_threshold, 2)),
            step=0.01
        )
    with c2:
        min_prec_note = assets["best_recall"]
        if min_prec_note is not None:
            st.metric("Recommended screening threshold", f"{assets['selected_threshold']:.2f}")

    if st.button("Run screening", type="primary"):
        prob, pred = make_prediction(user_input_df, assets, threshold)
        band = risk_band(prob, threshold)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted probability", f"{prob:.3f}")
        m2.metric("Screening decision", "Positive" if pred == 1 else "Negative")
        m3.metric("Risk band", band)

        st.progress(float(min(prob, 1.0)))

        if pred == 1:
            st.warning("This is a screening-positive result. It is not a diagnosis. Use it to trigger follow-up assessment.")
        else:
            st.success("This is a screening-negative result under the selected threshold.")

        local_fig, local_contrib = plot_local_shap(user_input_df, assets)
        st.pyplot(local_fig, use_container_width=True)
        st.dataframe(local_contrib, use_container_width=True)

        csv = user_input_df.assign(predicted_probability=prob, threshold=threshold, screening_prediction=pred)
        st.download_button(
            "Download this screening result",
            data=csv.to_csv(index=False).encode("utf-8"),
            file_name="screening_result.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Model performance")
    st.markdown("**Machine learning algorithm:** XGBoost Classifier")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Default threshold (0.50)**")
        st.dataframe(pd.DataFrame([assets["metrics_default"]]), use_container_width=True)

    with c2:
        st.markdown(f"**Selected screening threshold ({assets['selected_threshold']:.2f})**")
        st.dataframe(pd.DataFrame([assets["metrics_selected"]]), use_container_width=True)

    st.pyplot(plot_threshold_tradeoff(assets["thresholds_df"]), use_container_width=True)

    thr_show = assets["thresholds_df"].copy()
    thr_show = thr_show.round(4)
    st.dataframe(thr_show, use_container_width=True, height=320)

with tab3:
    st.subheader("SHAP insights")
    st.caption("These explanations are generated from the XGBoost model used for screening.")
    st.write("These plots explain which variables drive the model globally and how they influence individual screening results.")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_shap_bar(assets), use_container_width=True)
    with c2:
        st.dataframe(assets["shap_importance_df"].head(15), use_container_width=True)

    st.pyplot(plot_shap_summary(assets), use_container_width=True)

    top_features = assets["shap_importance_df"]["feature"].head(8).tolist()
    selected_feat = st.selectbox("Inspect a top driver", options=top_features)

    fig = plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        selected_feat,
        assets["shap_array"],
        assets["X_shap"],
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

with tab4:
    st.subheader("Dataset audit")
    st.write(f"Rows: {len(assets['df']):,}")
    st.write(f"Columns: {assets['df'].shape[1]}")
    st.write(f"Positive class rate: {assets['df'][TARGET].mean():.4f}")

    audit_df = pd.DataFrame({
        "column": assets["df"].columns,
        "dtype": [str(assets["df"][c].dtype) for c in assets["df"].columns],
        "missing": [int(assets["df"][c].isna().sum()) for c in assets["df"].columns],
        "sample_value": [assets["df"][c].iloc[0] for c in assets["df"].columns]
    })
    st.dataframe(audit_df, use_container_width=True, height=450)

    st.markdown("**Top SHAP drivers explained simply**")
    driver_table = assets["shap_importance_df"].head(10).copy()
    driver_table["label"] = driver_table["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    st.dataframe(driver_table[["feature", "label", "mean_abs_shap"]], use_container_width=True)
