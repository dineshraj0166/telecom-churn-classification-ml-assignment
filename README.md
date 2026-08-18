# Telecom Customer Churn Classification

## Student Information

- **Name:** Dinesh Raj R
- **BITS ID:** 2025ac05755
- **Programme:** M.Tech Artificial Intelligence and Machine Learning
- **Course:** Machine Learning
- **Assignment:** Assignment 2

## A. Problem Statement

Customer churn is an important concern for telecommunication companies because retaining an existing customer is generally more cost-effective than acquiring a new one. Identifying customers who are likely to discontinue their service can help a telecom company plan targeted retention measures.

The objective of this project is to develop and compare classification models that predict whether a telecom customer will churn based on complaints, subscription information, call activity, SMS usage, service status, age and customer value.

The target variable is `Churn`:

- `0` - The customer did not churn.
- `1` - The customer churned.

An interactive Streamlit application is also provided. It allows an evaluator to upload test data, select a trained model and view its predictions and evaluation results.

## B. Dataset Description

The project uses the **Iranian Telecom Customer Churn Dataset** from the UCI Machine Learning Repository.

- **Dataset:** Iranian Telecom Customer Churn
- **Source:** UCI Machine Learning Repository
- **Dataset URL:** [UCI Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
- **Number of instances:** 3,150
- **Number of input features:** 13
- **Target variable:** `Churn`
- **Problem type:** Binary classification
- **Missing values:** None

The dataset was collected from an Iranian telecommunications company's customer database over 12 months. The input attributes contain customer information aggregated during the first nine months. The churn label represents the customer's churn status at the end of the 12-month observation period, leaving a three-month planning gap.

### Input Features

1. Call Failure
2. Complains
3. Subscription Length
4. Charge Amount
5. Seconds of Use
6. Frequency of use
7. Frequency of SMS
8. Distinct Called Numbers
9. Age Group
10. Tariff Plan
11. Status
12. Age
13. Customer Value

The dataset satisfies the assignment requirement of at least 500 instances and 12 input features.

## C. GitHub Repository Link

**GitHub Repository:** [Telecom Churn Classification ML Assignment](https://github.com/dineshraj0166/telecom-churn-classification-ml-assignment)

The repository contains the source code, dependency file, README, test dataset, trained models, experimental notebook and evaluation results.

## D. Models Used and Evaluation Metrics

The following five classification models explicitly listed in the assignment were implemented on the same dataset and evaluated using the same test partition:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

Each model was evaluated using:

1. Accuracy
2. Area Under the ROC Curve (AUC)
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8968 | 0.9208 | 0.8400 | 0.4242 | 0.5638 | 0.5509 |
| Decision Tree | 0.9127 | 0.9077 | 0.7750 | 0.6263 | 0.6927 | 0.6475 |
| K-Nearest Neighbors | 0.9524 | 0.9677 | 0.8632 | 0.8283 | 0.8454 | 0.8175 |
| Gaussian Naive Bayes | 0.7381 | 0.8986 | 0.3648 | **0.8990** | 0.5190 | 0.4536 |
| Random Forest (Ensemble) | **0.9556** | **0.9875** | **0.8901** | 0.8182 | **0.8526** | **0.8275** |

## E. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved good accuracy, AUC and precision. However, its recall was only 0.4242, indicating that it missed a considerable number of customers who actually churned. |
| Decision Tree | The Decision Tree provided moderate and reasonably balanced performance. Its recall and F1 score were better than those of Logistic Regression, but its overall results were lower than KNN and Random Forest. |
| K-Nearest Neighbors | KNN achieved strong results across all metrics. Its recall of 0.8283 was slightly higher than Random Forest's recall, making it effective at detecting customers who actually churned. |
| Gaussian Naive Bayes | Gaussian Naive Bayes achieved the highest recall of 0.8990, identifying approximately 89.90% of actual churn customers. However, its low precision of 0.3648 indicates that it incorrectly classified many non-churn customers as churners. Its performance may have been affected by the strong correlation among several usage-related features. |
| Random Forest (Ensemble) | Random Forest achieved the highest Accuracy, AUC, Precision, F1 and MCC. It provided the most consistent and balanced performance while maintaining good churn-class recall. |
| **Overall Winner** | **Random Forest was selected as the overall winner because it achieved the best results in five of the six required evaluation metrics and provided the strongest balance between detecting churn customers and limiting incorrect churn predictions.** |

## F. Live Streamlit Application

**Live Application:** [Telecom Customer Churn Analytics](https://telecom-churn-ml-assignment.streamlit.app/)

The application allows an evaluator to upload the supplied `test_data.csv`, select any trained classification model and view its evaluation results.

## G. Data Preparation

The following data-preparation and exploratory-analysis steps were performed:

1. Loaded the dataset from the UCI Machine Learning Repository.
2. Examined the dataset dimensions, column types and descriptive statistics.
3. Checked for missing values and duplicate records.
4. Analysed the churn-class distribution.
5. Visualised numerical-feature distributions and their relationships with churn.
6. Examined feature correlations using a correlation heatmap.
7. Separated the input features and target variable.
8. Split the data into 80% training data and 20% test data.
9. Used stratified splitting to preserve the churn-class distribution.
10. Applied standard scaling inside pipelines for Logistic Regression, KNN and Gaussian Naive Bayes.
11. Used the same training and test partitions for every model to ensure a fair comparison.

The dataset was split using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Scaling was fitted only on the training data through model pipelines to avoid data leakage.

## H. Streamlit Application Features

The Streamlit application includes the following assignment requirements:

- Test-data upload in CSV format
- Model-selection dropdown
- Uploaded-data preview and validation
- Accuracy, AUC, Precision, Recall, F1 and MCC display
- Confusion matrix
- Classification report
- Precomputed comparison of all trained models
- Customer-level churn predictions and churn probabilities
- Downloadable prediction results

The uploaded CSV must contain the 13 original input features and the actual `Churn` column.

## I. Project Structure

```text
telecom-churn-classification-ml-assignment/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   `-- random_forest.pkl
|-- results/
|   |-- model_metrics.csv
|   |-- model_ranking.csv
|   `-- best_models_by_metric.csv
`-- notebook/
    `-- iranian_telecom_churn_classification.ipynb
```

## J. Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/dineshraj0166/telecom-churn-classification-ml-assignment.git
cd telecom-churn-classification-ml-assignment
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit application

```bash
streamlit run app.py
```

### 4. Evaluate a model

1. Upload `test_data.csv`.
2. Select a classification model.
3. Click **Evaluate Selected Model**.
4. Review the metrics, confusion matrix, classification report and predictions.

## K. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib
- Google Colab
- BITS Virtual Lab
- GitHub
- Streamlit Community Cloud

## L. Conclusion

The project demonstrated an end-to-end machine-learning workflow for telecom customer churn prediction, including data exploration, preprocessing, model development, comparative evaluation and interactive deployment. Random Forest produced the strongest overall performance, while Gaussian Naive Bayes produced the highest churn recall at the cost of many false-positive predictions.

