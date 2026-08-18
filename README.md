
# Telecom Customer Churn Classification

## Student Information

- **Name:** Dinesh Raj R
- **BITS ID:** 2025ac05755
- **Programme:** M.Tech Artificial Intelligence and Machine Learning
- **Course:** Machine Learning
- **Assignment:** Assignment 2

## A. Problem Statement

Customer churn is an important concern for telecommunication companies because acquiring a new customer is generally more expensive than retaining an existing customer.

The objective of this project is to develop and compare multiple machine-learning classification models that predict whether a telecom customer will churn based on customer complaints, subscription information, call activity, SMS usage, service status and customer value.

The project also provides an interactive Streamlit application through which test data can be uploaded and evaluated using different trained classification models.

The target variable is `Churn`:

- `0` represents a customer who did not churn.
- `1` represents a customer who churned.

## B. Dataset Description

The project uses the **Iranian Telecom Customer Churn Dataset** available from the UCI Machine Learning Repository.

- **Source:** UCI Machine Learning Repository
- **Dataset URL:** https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- **Number of instances:** 3,150
- **Number of input features:** 13
- **Target variable:** Churn
- **Problem type:** Binary classification
- **Missing values:** None

The dataset was collected from an Iranian telecommunication company's customer database over a period of 12 months. The input features contain customer information aggregated during the first nine months, while the churn label represents the customer’s status at the end of the twelve-month observation period.

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

## C. GitHub Repository Link

**GitHub Repository:**https://github.com/dineshraj0166/telecom-churn-classification-ml-assignment

## D. Live Streamlit Application

**Streamlit Application:** https://telecom-churn-ml-assignment.streamlit.app/

## E. Data Preparation

The following preprocessing steps were performed:

1. Loaded the dataset from the UCI Machine Learning Repository.
2. Examined dataset dimensions, column types and descriptive statistics.
3. Checked for missing values and duplicate records.
4. Analysed the churn-class distribution.
5. Visualised numerical-feature distributions and relationships with churn.
6. Examined feature correlations using a correlation heatmap.
7. Separated the input features and target variable.
8. Split the dataset into 80% training data and 20% test data.
9. Used stratified splitting to preserve the churn-class distribution.
10. Applied standard scaling inside pipelines for Logistic Regression, KNN and Gaussian Naive Bayes.
11. Used the same train/test split for all models to ensure a fair comparison.

The dataset was divided using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
