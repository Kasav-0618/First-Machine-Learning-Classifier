# Project Write-Up: First Machine Learning Classifier (Titanic Survival)

## 1. Project Overview & Objective
In this project, we built a supervised machine learning classification pipeline using **Scikit-Learn** to predict passenger survival on the Titanic. The goal was to establish a rigorous evaluation pipeline using a train/test split, train a Decision Tree classifier, evaluate it on unseen data using multiple performance metrics (Accuracy, Confusion Matrix, and Mean Squared Error), and compare it against a simple naive baseline.

---

## 2. Experimental Setup & Methodology

1. **Dataset & Preprocessing:**
   - **Features utilized:** Passenger Class (`Pclass`), Sex (`Sex`), Age (`Age`), Siblings/Spouses aboard (`SibSp`), Parents/Children aboard (`Parch`), Fare (`Fare`), and Port of Embarkation (`Embarked`).
   - Missing values in `Age` and `Fare` were imputed using the median to prevent skew from extreme values. Missing `Embarked` entries were imputed using the mode.
   - Categorical features were encoded (`Sex` mapped to binary `0/1`, `Embarked` one-hot encoded).

2. **Train / Test Split:**
   - The dataset was split into an **80% training set** and a **20% holdout test set**.
   - `stratify=y` was applied to ensure the proportion of survivors (~38%) and non-survivors (~62%) remained identical across both splits.
   - `random_state=42` was strictly fixed so all splits and model runs are 100% reproducible.

3. **Models Evaluated:**
   - **Simple Baseline:** A `DummyClassifier` with strategy `most_frequent` (predicts that all passengers perish).
   - **Unconstrained Decision Tree:** `DecisionTreeClassifier(max_depth=None)` to demonstrate overfitting.
   - **Regularized Decision Tree:** `DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)`.

---

## 3. Results & Baseline Comparison

| Model | Train Accuracy | Test Accuracy | Test MSE (Discrete) | Brier Score (Prob. MSE) |
| :--- | :---: | :---: | :---: | :---: |
| **Simple Baseline (Always 0)** | 61.66% | **61.45%** | **0.3855** | 0.3855 |
| **Unconstrained Tree (Overfit)** | 98.45% | 76.54% | 0.2346 | 0.2291 |
| **Pruned Tree (max_depth=3)** | 83.29% | **79.89%** | **0.2011** | **0.1482** |

### Key Observations:
- **Beating the Baseline:** The regularized Decision Tree achieved **~79.89% accuracy** on unseen test data, which is an **18.44 percentage point gain** over the naive majority-class baseline (61.45%).
- **Mean Squared Error (MSE):** The baseline had an MSE of 0.3855, whereas the pruned tree reduced the discrete MSE down to 0.2011 (and 0.1482 Brier probability error), confirming substantially higher calibration and prediction confidence.
- **Overfitting Demonstration:** The unconstrained tree scored over 98% on the training set but dropped sharply to 76.54% on test data (a ~22% generalization gap). Setting `max_depth=3` closed this gap and improved holdout generalization.

---

## 4. Confusion Matrix Analysis

On the 179 unseen test samples:
- **True Negatives (TN):** 97 (Passengers correctly predicted as perished)
- **False Positives (FP):** 13 (Passengers predicted to survive who actually perished)
- **False Negatives (FN):** 23 (Passengers who survived but were predicted to perish)
- **True Positives (TP):** 46 (Passengers correctly predicted as survived)

### Why Accuracy Alone Can Mislead:
Because 61.6% of passengers perished, a model predicting "everyone dies" obtains 61.45% accuracy while completely failing to identify a single survivor (0% recall on class 1). The Confusion Matrix reveals that our Decision Tree identifies **66.7% of all survivors** (Recall = 46/69) with a **Precision of 78.0%** (46/59).

---

## 5. What the Model Learned (Feature Importances)
Inspecting the tree splits highlights the historical reality of the disaster:
1. **Sex (Importance ~60-65%):** The primary root split. Females had a dramatically higher survival rate ("women and children first").
2. **Passenger Class (`Pclass`) & Fare (Importance ~20%):** Higher socio-economic status afforded better access to lifeboats on upper decks.
3. **Age & Family size (`SibSp` / `Parch`):** Younger children and small family groups had preferential evacuation.

---

## 6. Limitations of the Model & Dataset

1. **Small Sample Size:** The Titanic training set contains fewer than 1,000 samples. With only ~179 samples in the test split, each misclassified passenger shifts accuracy by ~0.56%, meaning metrics are sensitive to sample noise.
2. **Missing Feature Context:** Important factors like exact cabin location, proximity to exits, and crew instructions were unavailable or had heavy missingness (e.g. `Cabin` was >75% missing).
3. **Model Complexity (Single Tree):** A single decision tree relies on axis-aligned orthogonal splits. Ensemble techniques (e.g., Random Forests or Gradient Boosting) or regularized Logistic Regression could provide smoother probability estimates and higher robustness.
