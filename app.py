import base64
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

NAVY = "#0B1F4B"
CRIMSON = "#B5122E"
GREY = "#8A94A6"
LIGHT_BLUE = "#EEF3FA"
LIGHT_RED = "#FBECEE"
CARD_BORDER = "#7F9CCF"
BUTTON_HOVER = "#E3ECF9"
CHART_BLUE = "#0054A3"
AXIS_LINE = "#C9D3E3"
GRID_LINE = "#E8EEF7"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
BANNER_PATH = os.path.join(BASE_DIR, "assets", "Credit_card_fraud_detection.png")

plt.rcParams.update({
    "text.color": NAVY,
    "axes.labelcolor": NAVY,
    "axes.edgecolor": AXIS_LINE,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID_LINE,
    "axes.axisbelow": True,
    "legend.frameon": False,
})

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.markdown(
    f"""
    <style>
    [data-testid="stMetric"] {{
        border: 1px solid {CARD_BORDER} !important;
        box-shadow: 0 3px 10px rgba(11, 31, 75, 0.14);
        background-color: #FFFFFF;
    }}
    [data-testid="stTab"] p {{
        font-size: 18px !important;
        font-weight: 700 !important;
        color: {NAVY} !important;
    }}
    [data-testid="stBaseButton-secondary"] {{
        background-color: {NAVY} !important;
        color: #FFFFFF !important;
        border: 1px solid {NAVY} !important;
    }}
    [data-testid="stBaseButton-secondary"] p {{
        color: inherit !important;
    }}
    [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:active {{
        background-color: {BUTTON_HOVER} !important;
        color: {NAVY} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_deployment():
    model = tf.keras.models.load_model(os.path.join(ARTIFACT_DIR, "fraud_ann.keras"), compile=False)
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "ann_scaler.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "deployment_metadata.json")) as file:
        metadata = json.load(file)
    return model, scaler, metadata


@st.cache_data
def load_saved_results():
    comparison = pd.read_csv(os.path.join(ARTIFACT_DIR, "final_comparison.csv"))
    curves = pd.read_csv(os.path.join(ARTIFACT_DIR, "test_pr_curves.csv"))
    demo = pd.read_csv(os.path.join(ARTIFACT_DIR, "test_sample.csv"))
    return comparison, curves, demo


@st.cache_data
def load_banner(modified_time):
    with open(BANNER_PATH, "rb") as file:
        return base64.b64encode(file.read()).decode()


model, scaler, metadata = load_deployment()
comparison, curves, demo = load_saved_results()
feature_columns = metadata["feature_columns"]
threshold = metadata["threshold"]


def score_transactions(data):
    features = data[feature_columns].astype(float)
    scores = model.predict(scaler.transform(features), verbose=0).ravel()
    predictions = (scores >= threshold).astype(int)
    return scores, predictions


def outcome_label(actual, predicted):
    if actual == 1 and predicted == 1:
        return "Fraud caught (true positive)"
    if actual == 1 and predicted == 0:
        return "Fraud missed (false negative)"
    if actual == 0 and predicted == 1:
        return "False alarm (false positive)"
    return "Correctly cleared (true negative)"


def class_label(value):
    return "Fraud" if value == 1 else "Not fraud"


def validate_upload(data):
    if data.empty:
        return ["The file contains no transactions."]
    errors = []
    for column in feature_columns:
        if column not in data.columns:
            errors.append(f"Missing required column: {column}")
    for column in data.columns:
        if column not in feature_columns and column != "Class":
            errors.append(f"Unexpected column: {column}")
    if errors:
        return errors
    for column in feature_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            errors.append(f"Column {column} contains missing values.")
        elif values.isna().any():
            errors.append(f"Column {column} contains non-numeric values.")
        elif np.isinf(values).any():
            errors.append(f"Column {column} contains infinite values.")
    if "Class" in data.columns:
        class_values = pd.to_numeric(data["Class"], errors="coerce")
        if data["Class"].isna().any():
            errors.append("Column Class contains missing values.")
        elif not class_values.isin([0, 1]).all():
            errors.append("Column Class must contain only 0 (not fraud) or 1 (fraud).")
    return errors


def result_card(score, prediction):
    if prediction == 1:
        colour, background, title = CRIMSON, LIGHT_RED, "Potential fraud detected"
    else:
        colour, background, title = NAVY, LIGHT_BLUE, "No fraud detected"
    st.markdown(
        f"""
        <div style="border-left: 6px solid {colour}; background-color: {background}; padding: 1rem 1.25rem; border-radius: 6px; margin-bottom: 0.75rem;">
            <h3 style="color: {colour}; margin: 0 0 0.4rem 0;">{title}</h3>
            <p style="margin: 0;">Fraud score: <b>{score:.4f}</b> &nbsp;·&nbsp; Decision threshold: <b>{threshold:.4f}</b> &nbsp;·&nbsp; Prediction: <b>{class_label(prediction)}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_batch_results(data):
    scores, predictions = score_transactions(data)
    results = pd.DataFrame({"fraud_score": scores, "predicted_class": predictions})
    has_class = "Class" in data.columns
    if has_class:
        actual = pd.to_numeric(data["Class"]).astype(int).to_numpy()
        results["actual_class"] = actual
        results["prediction_outcome"] = [outcome_label(a, p) for a, p in zip(actual, predictions)]
    results = pd.concat([results, data[feature_columns].astype(float).reset_index(drop=True)], axis=1)

    st.markdown("#### Batch summary")
    summary = st.columns(4)
    summary[0].metric("Transactions analysed", len(results), border=True)
    summary[1].metric("Predicted fraud", int(predictions.sum()), border=True)
    summary[2].metric("Predicted not fraud", int((predictions == 0).sum()), border=True)
    if has_class:
        summary[3].metric("Correctly classified", f"{int((actual == predictions).sum())} / {len(actual)}", border=True)
        truth = st.columns(4)
        truth[0].metric("Actual frauds", int(actual.sum()), border=True)
        truth[1].metric("Fraud caught", int(((actual == 1) & (predictions == 1)).sum()), border=True)
        truth[2].metric("Fraud missed", int(((actual == 1) & (predictions == 0)).sum()), border=True)
        truth[3].metric("False alarms", int(((actual == 0) & (predictions == 1)).sum()), border=True)
    st.caption("These are results on this file only, not the official model test metrics (see Model Performance).")

    st.dataframe(results, hide_index=True, column_config={"fraud_score": st.column_config.NumberColumn(format="%.4f")})
    st.download_button("Download predictions CSV", results.to_csv(index=False), file_name="fraud_predictions.csv", mime="text/csv")


if os.path.exists(BANNER_PATH):
    st.markdown(
        f'<img src="data:image/png;base64,{load_banner(os.path.getmtime(BANNER_PATH))}" alt="Credit Card Fraud Detection" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;">',
        unsafe_allow_html=True,
    )
else:
    st.title("Credit Card Fraud Detection")
st.markdown("Detect potentially fraudulent credit-card transactions using the tuned ANN selected from the ML-vs-DL comparison.")

tab_detect, tab_batch, tab_performance = st.tabs(["Fraud Detection", "Batch Analysis", "Model Performance"])

with tab_detect:
    st.subheader("Test a demo transaction")
    st.caption(f"The demo transactions come from the held-out test set. {metadata['demo_sample_note']}")
    choice = st.selectbox("Select a demo transaction", range(len(demo)), format_func=lambda i: f"Demo Transaction {i + 1}")
    demo_row = demo.iloc[[choice]]

    if st.button("Predict Fraud", key="predict_demo"):
        scores, predictions = score_transactions(demo_row)
        result_card(scores[0], predictions[0])
        actual = int(demo_row["Class"].iloc[0])
        st.markdown(f"**Actual class:** {class_label(actual)} &nbsp;·&nbsp; **Outcome:** {outcome_label(actual, predictions[0])}")

    with st.expander("View transaction features"):
        st.dataframe(demo_row[feature_columns].T.rename(columns={demo_row.index[0]: "value"}), width="stretch")

    with st.expander("Advanced manual entry"):
        st.caption("The fields start with the values of the selected demo transaction. Time is the number of seconds elapsed since the first transaction in this dataset. V1–V28 are anonymised PCA components.")
        manual_values = {}
        field_columns = st.columns(5)
        for i, feature in enumerate(feature_columns):
            manual_values[feature] = field_columns[i % 5].number_input(feature, value=float(demo_row[feature].iloc[0]), format="%.6f", key=f"manual_{choice}_{feature}")

        if st.button("Predict manual transaction", key="predict_manual"):
            manual_row = pd.DataFrame([manual_values])
            scores, predictions = score_transactions(manual_row)
            result_card(scores[0], predictions[0])
            unchanged = all(np.isclose(manual_values[feature], demo_row[feature].iloc[0], atol=1e-6) for feature in feature_columns)
            if unchanged:
                actual = int(demo_row["Class"].iloc[0])
                st.markdown(f"**Actual class:** {class_label(actual)} &nbsp;·&nbsp; **Outcome:** {outcome_label(actual, predictions[0])}")
            else:
                st.caption("The values differ from the demo transaction, so no actual class is available for this input.")

with tab_batch:
    st.subheader("Analyse many transactions")
    source = st.radio("Choose input source", ["Use demo CSV", "Upload CSV"], horizontal=True)
    batch = None

    if source == "Use demo CSV":
        st.info(f"This demonstration dataset contains {len(demo)} transactions: {int(demo['Class'].sum())} fraud and {int((demo['Class'] == 0).sum())} non-fraud. {metadata['demo_sample_note']}")
        buttons = st.columns(2)
        analyse = buttons[0].button("Analyse demo transactions")
        buttons[1].download_button("Download demo CSV", demo.to_csv(index=False), file_name="test_sample.csv", mime="text/csv")
        if analyse:
            batch = demo
    else:
        st.caption("Required columns: Time, V1–V28 and Amount, in any order. Optional column: Class (0 = not fraud, 1 = fraud), used only to compare predictions with the actual class. No other columns are accepted.")
        uploaded = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded is not None:
            try:
                data = pd.read_csv(uploaded)
            except Exception:
                data = None
                st.error("The file could not be read as a CSV.")
            if data is not None:
                errors = validate_upload(data)
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    batch = data

    if batch is not None:
        show_batch_results(batch)

with tab_performance:
    deployed = comparison[comparison["model"] == metadata["model_name"]].iloc[0]
    runner_up = comparison[comparison["model"] != metadata["model_name"]].iloc[0]
    ann_matrix = json.loads(deployed["test_confusion_matrix"])
    rf_matrix = json.loads(runner_up["test_confusion_matrix"])
    ann_tn, ann_fp, ann_fn, ann_tp = ann_matrix[0][0], ann_matrix[0][1], ann_matrix[1][0], ann_matrix[1][1]
    rf_tn, rf_fp, rf_fn, rf_tp = rf_matrix[0][0], rf_matrix[0][1], rf_matrix[1][0], rf_matrix[1][1]
    total_frauds = ann_tp + ann_fn
    total_transactions = ann_tn + ann_fp + ann_fn + ann_tp
    ann_label = "Tuned ANN"
    rf_label = runner_up["model"]

    st.subheader("Why this model?")
    st.markdown(
        "The practical objective of this project is to detect as many fraudulent transactions as possible. "
        "Recall is therefore especially important when interpreting the final deployment choice. "
        "PR-AUC is also used because it evaluates the precision–recall trade-off on this highly imbalanced dataset."
    )
    st.markdown(
        f"On the held-out test set, the ANN detected **{ann_tp} of {total_frauds}** fraud transactions, compared with **{rf_tp}** for {rf_label}, "
        f"and achieved a slightly higher PR-AUC ({deployed['test_pr_auc']:.3f} vs {runner_up['test_pr_auc']:.3f}). "
        f"{rf_label} raised fewer false alarms ({rf_fp} vs {ann_fp}) and achieved higher precision and F1, "
        "so the choice represents an operating trade-off rather than the ANN being better on every metric."
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("#### Final ML vs DL metrics (test set)")
        metric_names = ["PR-AUC", "Precision", "Recall", "F1", "ROC-AUC"]
        metric_columns = ["test_pr_auc", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
        positions = np.arange(len(metric_names))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ann_bars = ax.bar(positions - 0.2, [deployed[c] for c in metric_columns], 0.4, label=ann_label, color=NAVY)
        rf_bars = ax.bar(positions + 0.2, [runner_up[c] for c in metric_columns], 0.4, label=rf_label, color=CHART_BLUE)
        ax.bar_label(ann_bars, fmt="%.3f", fontsize=8)
        ax.bar_label(rf_bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(positions, metric_names)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score")
        ax.legend(loc="upper left", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            f"**What the chart says:** The ANN has the higher recall ({deployed['test_recall']:.3f} vs {runner_up['test_recall']:.3f}) "
            f"and a slightly higher PR-AUC ({deployed['test_pr_auc']:.3f} vs {runner_up['test_pr_auc']:.3f}), so it catches more fraud and performs marginally better across the precision–recall curve. "
            f"{rf_label} has higher precision ({runner_up['test_precision']:.3f} vs {deployed['test_precision']:.3f}), F1 ({runner_up['test_f1']:.3f} vs {deployed['test_f1']:.3f}) "
            f"and ROC-AUC ({runner_up['test_roc_auc']:.3f} vs {deployed['test_roc_auc']:.3f}). "
            "Since the objective is to detect more fraud, the ANN's recall advantage is particularly important, while its lower precision shows the cost of that choice."
        )

    with chart_right:
        st.markdown("#### Test precision–recall curves")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for row, colour, label in [(deployed, NAVY, ann_label), (runner_up, CHART_BLUE, rf_label)]:
            curve = curves[curves["model"] == row["model"]]
            ax.plot(curve["recall"], curve["precision"], drawstyle="steps-post", color=colour, label=f"{label} (PR-AUC {row['test_pr_auc']:.3f})")
            ax.plot(row["test_recall"], row["test_precision"], marker="o", markersize=8, color=colour)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            f"**What the chart says:** The two models perform very similarly across the full range of thresholds. "
            f"The ANN's overall test PR-AUC is {deployed['test_pr_auc']:.3f} compared with {runner_up['test_pr_auc']:.3f} for {rf_label}, a difference of only "
            f"{deployed['test_pr_auc'] - runner_up['test_pr_auc']:.3f}, so the ANN is a narrow winner rather than clearly superior. "
            "The dots mark where each model operates at its chosen decision threshold."
        )

    st.markdown("#### Fraud detection trade-off (test set)")
    outcome_names = ["Fraud caught", "Fraud missed", "False alarms"]
    positions = np.arange(len(outcome_names))
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ann_bars = ax.bar(positions - 0.2, [ann_tp, ann_fn, ann_fp], 0.4, label=ann_label, color=NAVY)
    rf_bars = ax.bar(positions + 0.2, [rf_tp, rf_fn, rf_fp], 0.4, label=rf_label, color=CHART_BLUE)
    ax.bar_label(ann_bars, fontsize=10)
    ax.bar_label(rf_bars, fontsize=10)
    ax.set_xticks(positions, outcome_names)
    ax.set_ylabel("Number of transactions")
    ax.set_ylim(0, max(ann_tp, rf_tp) * 1.2)
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        f"**What the chart says:** The ANN catches **{ann_tp - rf_tp} more** fraud transactions than {rf_label} ({ann_tp} vs {rf_tp} of {total_frauds}), "
        "which directly supports the objective of detecting as much fraud as possible. "
        f"The trade-off is **{ann_fp - rf_fp} more** false alarms ({ann_fp} vs {rf_fp}). "
        f"{rf_label} is therefore more conservative, while the ANN prioritises catching additional fraud."
    )

    st.markdown("#### Comparison table (test set)")
    table = pd.DataFrame(
        {
            ann_label: [f"{deployed[c]:.4f}" for c in metric_columns] + [f"{ann_tp} / {total_frauds}", str(ann_fn), str(ann_fp)],
            rf_label: [f"{runner_up[c]:.4f}" for c in metric_columns] + [f"{rf_tp} / {total_frauds}", str(rf_fn), str(rf_fp)],
        },
        index=metric_names + ["Fraud caught", "Fraud missed", "False alarms"],
    )
    st.table(table)

    info_left, info_middle, info_right = st.columns(3)
    info_left.info(
        f"**Why not accuracy?** Only {total_frauds} of the {total_transactions:,} test transactions ({total_frauds / total_transactions:.2%}) are fraud. "
        f"Predicting every transaction as not fraud would therefore be {(total_transactions - total_frauds) / total_transactions:.2%} accurate while detecting no fraud at all, "
        "so accuracy alone is misleading. Precision, recall, F1 and PR-AUC give a much more meaningful picture."
    )
    info_middle.info(
        f"**Decision threshold:** The deployed threshold ({threshold:.4f}) was selected by maximising validation F1 because no business cost ratio, target recall or acceptable false-positive rate was defined. "
        "This creates a balanced operating point rather than explicitly maximising recall. A recall-prioritised threshold would produce a different fraud-detection / false-alarm trade-off."
    )
    info_right.info(
        "**Feature interpretation:** V1–V28 are anonymised PCA components, and the deployed ANN is a nonlinear model. "
        "The project therefore does not assign business meanings to individual features or present ANN weights as feature importance."
    )

    with st.expander("Dataset and model limitations"):
        st.markdown(
            """
- Only 71 fraud cases were present in each of the validation and test sets, so model differences are based on relatively few positive examples.
- The validation set was used for model selection, ANN early stopping and threshold tuning, so validation scores are optimistic.
- `Time` only has meaning relative to the first transaction in this dataset.
- `V1`–`V28` are anonymised PCA components and cannot be interpreted in business terms.
- 1,081 exact duplicate rows were removed before splitting, preventing identical observations from being duplicated across dataset partitions. Without transaction IDs, it could not be established whether these represented data errors or genuine repeated transactions.
- 48 rows with unusual exact whole-number PCA values were kept because their validity could not be established.
- The dataset covers two days of transactions by European cardholders in September 2013, so results should not be assumed to generalise to current real-world transaction streams.
            """
        )

st.markdown(
    f"""
    <hr style="margin-top: 2rem;">
    <p style="color: {GREY}; font-size: 0.8rem; text-align: center;">
    Deployment model: Tuned ANN · Threshold: {threshold:.4f} · Inputs: 30 features (Time, V1–V28, Amount) · Primary development metric: PR-AUC · Practical deployment focus: Recall · V1–V28 are anonymised PCA components
    </p>
    """,
    unsafe_allow_html=True,
)
