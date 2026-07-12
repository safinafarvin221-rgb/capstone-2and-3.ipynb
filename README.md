# Telecom Churn Capstone Project – Part 2 & Part 3

## Project Overview

This project focuses on building, evaluating, and comparing supervised machine learning models using the cleaned telecom customer dataset prepared in Part 1.

The project includes regression, classification, ensemble learning, hyperparameter tuning, feature importance analysis, feature ablation, cross-validation, learning curve analysis, and model serialization.

---

## Dataset

Dataset Used:
- cleaned_data.csv (Generated in Capstone Part 1)

Regression Target:
- monthly_charges

Classification Target:
- A binary target created by comparing monthly_charges with its median.

```python
y_classification = (monthly_charges > monthly_charges.median()).astype(int)
```

---

# Data Preprocessing

The following preprocessing steps were performed:

- Loaded cleaned_data.csv
- Created regression and classification targets
- Converted signup_date into Year and Month features
- Removed unnecessary columns
- Applied One-Hot Encoding to categorical columns
- Dropped the first dummy column to avoid multicollinearity
- Split the dataset using Train-Test Split (80:20)
- Applied SimpleImputer (Median Strategy)
- Applied StandardScaler where required
- Used SMOTE to handle class imbalance for Logistic Regression

The scaler was fitted only on the training data to prevent data leakage.

---

# Part 2 – Supervised Machine Learning

## Regression Models

### Linear Regression

Model Evaluation

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The coefficients of all features were printed and interpreted.

Large Positive Coefficient:
- Increasing the feature increases the predicted monthly charges.

Large Negative Coefficient:
- Increasing the feature decreases the predicted monthly charges.

---

### Ridge Regression

A Ridge Regression model was trained using:

```python
Ridge(alpha=1.0)
```

Comparison metrics:

- MAE
- RMSE
- R² Score

Ridge regression applies L2 regularization to reduce overfitting and stabilize model coefficients.

---

## Logistic Regression

Performed Binary Classification using Logistic Regression.

Class imbalance was handled using **SMOTE**.

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- ROC-AUC Score

### Precision Formula

Precision = TP / (TP + FP)

### Recall Formula

Recall = TP / (TP + FN)

For this telecom churn prediction task, Recall is an important metric because missing customers who are likely to churn (False Negatives) can result in customer loss.

The ROC-AUC score measures the model's ability to distinguish between the two classes. A higher AUC indicates better classification performance.

---

# Part 3 – Advanced Machine Learning

## Decision Tree

Two Decision Tree models were trained.

### Default Decision Tree

Measured:

- Training Accuracy
- Test Accuracy

The default tree showed signs of overfitting because training accuracy was significantly higher than test accuracy.

---

### Controlled Decision Tree

Hyperparameters:

- max_depth = 5
- min_samples_split = 20

Reducing the depth helped improve generalization by reducing variance.

---

## Gini vs Entropy

Both splitting criteria were compared.

### Gini Impurity

Gini = 1 − Σ(pi²)

### Entropy

Entropy = −Σ(pi log₂(pi))

A Gini value of zero indicates that all samples in a node belong to the same class.

---

## Random Forest

Random Forest was trained using:

- n_estimators = 100
- max_depth = 10
- random_state = 42

Evaluation:

- Training Accuracy
- Test Accuracy
- ROC-AUC

The top five important features were identified using feature importance scores.

Random Forest computes feature importance based on the average reduction in Gini impurity across all trees.

Bagging improves performance by combining multiple decision trees trained on different bootstrap samples, reducing model variance.

---

## Gradient Boosting

Gradient Boosting model was trained using:

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3

Evaluation:

- Training Accuracy
- Test Accuracy
- ROC-AUC

---

## Feature Ablation

The five least important features were removed.

A second Random Forest model was trained using the reduced feature set.

The ROC-AUC scores of the full model and reduced model were compared to determine whether the removed features contributed meaningful information.

---

## Cross Validation

Performed 5-Fold Stratified Cross Validation.

Models Compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Cross-validation provides a more reliable estimate of model performance because every sample is used for both training and validation.

---

## Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest model.

Parameter Grid:

- Number of Trees
- Maximum Depth
- Minimum Samples per Leaf

The best hyperparameters and best cross-validation ROC-AUC score were reported.

---

## Learning Curve

The best pipeline was trained using:

- 20%
- 40%
- 60%
- 80%
- 100%

of the training dataset.

Training AUC and Test AUC were compared.

The learning curve helps determine whether model performance is limited by data size or model complexity.

---

## Model Serialization

The best pipeline obtained from GridSearchCV was saved using:

- joblib.dump()

The saved model was successfully reloaded using:

- joblib.load()

Predictions were generated successfully using the loaded model.

---

# Libraries Used

- pandas
- numpy
- matplotlib
- scikit-learn
- imbalanced-learn
- joblib

---

# Files Included

- capstone 2and 3.ipynb
- cleaned_data.csv
- best_model.pkl
- README.md

---

# Conclusion

Multiple regression and classification models were developed and compared.

Hyperparameter tuning improved model selection.

Feature importance and feature ablation helped identify influential variables.

Cross-validation and learning curve analysis improved confidence in model performance.

The best model was saved using Joblib for future prediction and deployment.
