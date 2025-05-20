# Machine Learning Regression and Classification Project

## 📌 Overview

This project demonstrates the application of various machine learning models for both **regression** and **classification** tasks. It involves importing a dataset, preprocessing the data, and applying models such as **Random Forest**, **XGBoost**, **LightGBM**, **Support Vector Machines**, and **Linear/Logistic Regression**. Model performance is evaluated using a variety of metrics.

## 📂 Dataset

* The dataset is provided in CSV format (uploaded as part of the project).
* It is loaded and processed using **Pandas**.
* Feature scaling is performed using `StandardScaler` or `MinMaxScaler`.

## 🛠️ Libraries Used

The following Python libraries are used in the project:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn import metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, ConfusionMatrixDisplay
)

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
```

## ⚙️ Project Workflow

1. **Data Loading**
   Load the dataset using `pandas.read_csv()`.

2. **Preprocessing**

   * Handle missing values (if any).
   * Feature scaling using `StandardScaler` or `MinMaxScaler`.

3. **Train-Test Split**
   The data is split into training and testing sets using `train_test_split`.

4. **Model Training**
   Different models are trained for:

   * **Regression:** Linear Regression, Random Forest Regressor, XGBoost Regressor, SVR, LightGBM
   * **Classification:** Logistic Regression, Random Forest Classifier, XGBoost Classifier, SVC

5. **Model Evaluation**

   * **Regression Metrics:** MAE, MSE, RMSE, R²
   * **Classification Metrics:** Accuracy, Precision, Recall, F1 Score, ROC AUC, Confusion Matrix

## 📊 Visualization

* Data visualization is done using **Matplotlib** and **Seaborn**.
* Confusion matrix and performance metrics are plotted for better interpretability.

## 📁 File Structure

```
project/
│
├── notebook.ipynb              # Main Jupyter notebook
├── dataset.csv                 # Input dataset
└── README.md                   # Project description
```

## 📌 Conclusion

This project showcases a comprehensive end-to-end pipeline for supervised learning tasks. It can serve as a solid foundation for expanding into more advanced feature engineering, model optimization (GridSearchCV, RandomizedSearchCV), and deployment.
