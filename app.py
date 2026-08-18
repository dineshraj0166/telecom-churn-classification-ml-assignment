
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📡",
    layout="wide"
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            color: #123B5D;
            font-size: 2.4rem;
            font-weight: 750;
            margin-bottom: 0;
        }

        .subtitle {
            color: #526777;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            color: #123B5D;
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 1rem;
        }

        [data-testid="stMetric"] {
            background-color: #F3F8FC;
            border: 1px solid #D7E6F2;
            border-radius: 10px;
            padding: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Application header
# ---------------------------------------------------------

st.markdown(
    '<p class="main-title">📡 Telecom Customer Churn Analytics</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="subtitle">
        Evaluate trained classification models using uploaded telecom
        customer test data.
    </p>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/knn.pkl",
    "Gaussian Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_comparison_results():
    results_path = "results/model_metrics.csv"

    if os.path.exists(results_path):
        return pd.read_csv(results_path)

    return None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Model Controls")

    selected_model_name = st.selectbox(
        "Select a classification model",
        options=list(MODEL_FILES.keys())
    )

    st.info(
        "The uploaded CSV must contain the original input features "
        "and the actual `Churn` column."
    )

    st.markdown("---")

    st.markdown(
        """
        **Target definition**

        - `0` — No Churn
        - `1` — Churn
        """
    )


# ---------------------------------------------------------
# Dataset uploader
# ---------------------------------------------------------

st.markdown(
    '<p class="section-title">Upload Test Dataset</p>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload the test data in CSV format",
    type=["csv"]
)


# ---------------------------------------------------------
# Model comparison
# ---------------------------------------------------------

comparison_results = load_comparison_results()

if comparison_results is not None:
    with st.expander(
        "View precomputed model comparison",
        expanded=False
    ):
        st.dataframe(
            comparison_results.round(4),
            use_container_width=True,
            hide_index=True
        )

        if "F1" in comparison_results.columns:
            best_model_row = comparison_results.loc[
                comparison_results["F1"].idxmax()
            ]

            st.success(
                "Best model based on F1 score: "
                f"{best_model_row['ML Model Name']} "
                f"({best_model_row['F1']:.4f})"
            )


# ---------------------------------------------------------
# Uploaded-data processing
# ---------------------------------------------------------

if uploaded_file is None:
    st.info(
        "Upload `test_data.csv` to begin model evaluation."
    )

else:
    try:
        test_data = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset uploaded successfully: "
            f"{test_data.shape[0]} rows and "
            f"{test_data.shape[1]} columns."
        )

        with st.expander(
            "Preview uploaded test data",
            expanded=True
        ):
            st.dataframe(
                test_data.head(20),
                use_container_width=True
            )

        if "Churn" not in test_data.columns:
            st.error(
                "The uploaded file does not contain the required "
                "`Churn` target column."
            )

            st.stop()

        if test_data.empty:
            st.error(
                "The uploaded CSV does not contain any records."
            )

            st.stop()

        model_path = MODEL_FILES[selected_model_name]

        if not os.path.exists(model_path):
            st.error(
                f"Model file is missing: `{model_path}`"
            )

            st.stop()

        selected_model = load_model(model_path)

        if hasattr(selected_model, "feature_names_in_"):
            expected_features = list(
                selected_model.feature_names_in_
            )
        else:
            expected_features = [
                column
                for column in test_data.columns
                if column != "Churn"
            ]

        uploaded_features = [
            column
            for column in test_data.columns
            if column != "Churn"
        ]

        missing_features = [
            feature
            for feature in expected_features
            if feature not in uploaded_features
        ]

        extra_features = [
            feature
            for feature in uploaded_features
            if feature not in expected_features
        ]

        if missing_features:
            st.error(
                "Required feature columns are missing: "
                + ", ".join(missing_features)
            )

            st.stop()

        if extra_features:
            st.warning(
                "Extra columns were ignored: "
                + ", ".join(extra_features)
            )

        X_uploaded = test_data[expected_features].copy()
        y_uploaded = test_data["Churn"].copy()

        if X_uploaded.isnull().any().any():
            st.error(
                "The uploaded input features contain missing values."
            )

            st.stop()

        if y_uploaded.isnull().any():
            st.error(
                "The `Churn` column contains missing values."
            )

            st.stop()

        invalid_targets = set(
            y_uploaded.unique()
        ) - {0, 1}

        if invalid_targets:
            st.error(
                "The `Churn` column must contain only 0 and 1. "
                f"Invalid values found: {invalid_targets}"
            )

            st.stop()

        non_numeric_columns = (
            X_uploaded
            .select_dtypes(exclude=np.number)
            .columns
            .tolist()
        )

        if non_numeric_columns:
            st.error(
                "The following features must be numeric: "
                + ", ".join(non_numeric_columns)
            )

            st.stop()

        st.markdown(
            '<p class="section-title">Model Evaluation</p>',
            unsafe_allow_html=True
        )

        st.write(
            f"Selected model: **{selected_model_name}**"
        )

        if st.button(
            "Evaluate Selected Model",
            type="primary",
            use_container_width=True
        ):
            y_prediction = selected_model.predict(
                X_uploaded
            )

            y_probability = selected_model.predict_proba(
                X_uploaded
            )[:, 1]

            accuracy = accuracy_score(
                y_uploaded,
                y_prediction
            )

            auc = roc_auc_score(
                y_uploaded,
                y_probability
            )

            precision = precision_score(
                y_uploaded,
                y_prediction,
                zero_division=0
            )

            recall = recall_score(
                y_uploaded,
                y_prediction,
                zero_division=0
            )

            f1 = f1_score(
                y_uploaded,
                y_prediction,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_uploaded,
                y_prediction
            )

            metric_columns = st.columns(6)

            metric_columns[0].metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            metric_columns[1].metric(
                "AUC",
                f"{auc:.4f}"
            )

            metric_columns[2].metric(
                "Precision",
                f"{precision:.4f}"
            )

            metric_columns[3].metric(
                "Recall",
                f"{recall:.4f}"
            )

            metric_columns[4].metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            metric_columns[5].metric(
                "MCC",
                f"{mcc:.4f}"
            )

            left_column, right_column = st.columns(2)

            with left_column:
                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_uploaded,
                    y_prediction,
                    labels=[0, 1]
                )

                figure, axis = plt.subplots(
                    figsize=(6, 4)
                )

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=[
                        "No Churn",
                        "Churn"
                    ],
                    yticklabels=[
                        "No Churn",
                        "Churn"
                    ],
                    ax=axis
                )

                axis.set_xlabel("Predicted Class")
                axis.set_ylabel("Actual Class")
                axis.set_title(
                    f"{selected_model_name} Confusion Matrix"
                )

                figure.tight_layout()

                st.pyplot(figure)

                plt.close(figure)

            with right_column:
                st.subheader("Classification Report")

                report = classification_report(
                    y_uploaded,
                    y_prediction,
                    target_names=[
                        "No Churn",
                        "Churn"
                    ],
                    output_dict=True,
                    zero_division=0
                )

                report_dataframe = (
                    pd.DataFrame(report)
                    .transpose()
                    .round(4)
                )

                st.dataframe(
                    report_dataframe,
                    use_container_width=True
                )

            prediction_results = test_data.copy()

            prediction_results[
                "Predicted Churn"
            ] = y_prediction

            prediction_results[
                "Churn Probability"
            ] = y_probability.round(4)

            st.subheader("Prediction Results")

            st.dataframe(
                prediction_results,
                use_container_width=True
            )

            prediction_csv = (
                prediction_results
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label="Download Prediction Results",
                data=prediction_csv,
                file_name=(
                    "telecom_churn_predictions.csv"
                ),
                mime="text/csv"
            )

    except pd.errors.EmptyDataError:
        st.error(
            "The uploaded CSV file is empty or invalid."
        )

    except Exception as error:
        st.error(
            "The application could not process the uploaded file."
        )

        st.exception(error)
