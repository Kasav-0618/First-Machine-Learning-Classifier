"""
Titanic Survival Prediction - First Machine Learning Classifier
Author: Machine Learning Project
Goal: Train a Decision Tree Classifier to predict Titanic passenger survival,
      evaluate on unseen test data, and compare against a simple baseline.

Key Concepts Covered:
- Train / Test Split (preventing data leakage and overfitting)
- Baseline Model Comparison (Dummy / Majority Class Classifier)
- Scikit-Learn Decision Tree Classifier
- Evaluation Metrics: Accuracy, Confusion Matrix, Mean Squared Error (MSE)
- Overfitting Analysis (Train vs. Test Performance)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    mean_squared_error
)

# ---------------------------------------------------------------------------
# 1. DATA LOADING & INSPECTION
# ---------------------------------------------------------------------------
def load_titanic_dataset():
    """
    Loads the Titanic dataset.
    Attempts to read from local 'titanic.csv' first, then seaborn, then online GitHub mirror.
    """
    local_csv = "titanic.csv"
    if os.path.exists(local_csv):
        print(f"Loading local dataset from: {local_csv}")
        df = pd.read_csv(local_csv)
    else:
        print("Fetching dataset online...")
        try:
            # Standard Kaggle/OpenML Titanic dataset URL
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
            df.to_csv(local_csv, index=False)
            print(f"Dataset downloaded and cached locally as '{local_csv}'.")
        except Exception as e:
            print(f"Could not download dataset from GitHub: {e}")
            print("Attempting to load via seaborn...")
            df = sns.load_dataset("titanic")
            # Rename columns to match standard schema if needed
            if "survived" in df.columns and "Survived" not in df.columns:
                df = df.rename(columns={"survived": "Survived", "pclass": "Pclass", "sex": "Sex", "age": "Age", "fare": "Fare", "embarked": "Embarked"})
    
    return df

print("=" * 70)
print("STEP 1: LOADING & EXPLORING DATA")
print("=" * 70)
df = load_titanic_dataset()
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 records:")
print(df.head())

# Target distribution
target_counts = df['Survived'].value_counts()
print("\nTarget Class Distribution ('Survived'):")
print(f"  Did Not Survive (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0) / len(df) * 100:.2f}%)")
print(f"  Survived (1)        : {target_counts.get(1, 0)} ({target_counts.get(1, 0) / len(df) * 100:.2f}%)")


# ---------------------------------------------------------------------------
# 2. FEATURE PREPROCESSING & CLEANING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING & DATA CLEANING")
print("=" * 70)

# Select relevant predictive features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Ensure required columns exist
available_features = [col for col in features if col in df.columns]
data = df[available_features + [target]].copy()

# Handle missing values:
# 1. Fill missing Age with median
median_age = data['Age'].median()
data['Age'] = data['Age'].fillna(median_age)

# 2. Fill missing Fare with median (if any)
if 'Fare' in data.columns:
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# 3. Fill missing Embarked with mode
if 'Embarked' in data.columns:
    mode_embarked = data['Embarked'].mode()[0]
    data['Embarked'] = data['Embarked'].fillna(mode_embarked)

# Encode categorical variables:
# Binary mapping for Sex: male -> 0, female -> 1
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for Embarked (C, Q, S)
if 'Embarked' in data.columns:
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

X = data.drop(columns=[target])
y = data[target].astype(int)

print(f"Features used for training: {list(X.columns)}")
print(f"Cleaned feature matrix shape: {X.shape}")


# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (FIXED RANDOM STATE)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: TRAIN / TEST SPLIT (80% Train, 20% Unseen Test)")
print("=" * 70)

# Golden Rule: Never evaluate a model on data it trained on!
# Fixed random_state ensures reproducibility across runs.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size    : {X_test.shape[0]} samples (unseen holdout data)")


# ---------------------------------------------------------------------------
# 4. SIMPLE BASELINE MODEL (MAJORITY CLASS)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: TRAINING SIMPLE BASELINE MODEL")
print("=" * 70)

# A simple baseline predicts the most frequent class (class 0: perish).
# If a complex model cannot beat this baseline, it offers no real predictive value!
baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_mse = mean_squared_error(y_test, y_pred_baseline)

print(f"Baseline Strategy: Always predict majority class ('Did Not Survive' / 0)")
print(f"Baseline Test Accuracy : {baseline_acc:.4f} ({baseline_acc * 100:.2f}%)")
print(f"Baseline Test MSE      : {baseline_mse:.4f}")


# ---------------------------------------------------------------------------
# 5. DECISION TREE CLASSIFIER (PRUNED VS UNCONSTRAINED / OVERFITTING)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: TRAINING SCIKIT-LEARN DECISION TREE CLASSIFIER")
print("=" * 70)

# A) Overfitted Decision Tree (No depth limit) to illustrate the danger of overfitting
tree_unconstrained = DecisionTreeClassifier(random_state=42)
tree_unconstrained.fit(X_train, y_train)

train_acc_unconstrained = accuracy_score(y_train, tree_unconstrained.predict(X_train))
test_acc_unconstrained = accuracy_score(y_test, tree_unconstrained.predict(X_test))

print("A) Unconstrained Tree (max_depth=None):")
print(f"   Train Accuracy : {train_acc_unconstrained:.4f} ({train_acc_unconstrained * 100:.2f}%)  <-- Memorized data!")
print(f"   Test Accuracy  : {test_acc_unconstrained:.4f} ({test_acc_unconstrained * 100:.2f}%)   <-- Large generalization gap")

# B) Regularized / Pruned Decision Tree (max_depth=3)
tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

train_acc_pruned = accuracy_score(y_train, tree_model.predict(X_train))
test_acc_pruned = accuracy_score(y_test, tree_model.predict(X_test))

print("\nB) Pruned Regularized Tree (max_depth=3):")
print(f"   Train Accuracy : {train_acc_pruned:.4f} ({train_acc_pruned * 100:.2f}%)")
print(f"   Test Accuracy  : {test_acc_pruned:.4f} ({test_acc_pruned * 100:.2f}%)  <-- Better generalization!")


# ---------------------------------------------------------------------------
# 6. EVALUATION METRICS ON UNSEEN TEST DATA
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: EVALUATION METRICS ON UNSEEN TEST SET")
print("=" * 70)

y_pred = tree_model.predict(X_test)
y_pred_proba = tree_model.predict_proba(X_test)[:, 1]

# 1. Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"1. Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# 2. Mean Squared Error (MSE) / Brier Score
# Compares actual binary outcomes (0 or 1) against predicted values
test_mse_discrete = mean_squared_error(y_test, y_pred)
test_mse_prob = mean_squared_error(y_test, y_pred_proba)
print(f"2. Mean Squared Error (Discrete predictions): {test_mse_discrete:.4f}")
print(f"   Mean Squared Error (Probability Brier Score): {test_mse_prob:.4f}")

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\n3. Confusion Matrix:")
print(f"               Predicted Perished (0)    Predicted Survived (1)")
print(f"Actual 0 (No)  TN = {tn:<24} FP = {fp}")
print(f"Actual 1 (Yes) FN = {fn:<24} TP = {tp}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Perished (0)", "Survived (1)"]))

# Comparison Summary
print("-" * 70)
print(f"{'Model':<28} | {'Accuracy':<10} | {'MSE':<10} | {'Delta vs Baseline'}")
print("-" * 70)
print(f"{'Dummy Baseline (All 0)':<28} | {baseline_acc * 100:>7.2f}%   | {baseline_mse:>8.4f} | {'Baseline'}")
print(f"{'Decision Tree (depth=3)':<28} | {test_accuracy * 100:>7.2f}%   | {test_mse_discrete:>8.4f} | {f'+{(test_accuracy - baseline_acc)*100:.2f}%'}")
print("-" * 70)


# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE & VISUALIZATIONS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: FEATURE IMPORTANCE & SAVING VISUALIZATIONS")
print("=" * 70)

importances = pd.Series(tree_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:")
for feat, imp in importances.items():
    print(f"  - {feat:<15}: {imp:.4f} ({imp * 100:.1f}%)")

# Plot 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Perished (0)", "Survived (1)"])
disp.plot(cmap="Blues", values_format="d", ax=ax)
ax.set_title("Decision Tree Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
plt.tight_layout()
cm_plot_path = "confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=150)
print(f"\n[Saved] Confusion matrix figure saved to '{cm_plot_path}'.")
plt.close()

# Plot 2: Decision Tree Diagram
fig, ax = plt.subplots(figsize=(16, 9))
plot_tree(
    tree_model,
    feature_names=X.columns,
    class_names=["Perished", "Survived"],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
ax.set_title("Trained Decision Tree (max_depth=3)", fontsize=14, fontweight="bold")
plt.tight_layout()
tree_plot_path = "decision_tree.png"
plt.savefig(tree_plot_path, dpi=150)
print(f"[Saved] Decision tree visualization saved to '{tree_plot_path}'.")
plt.close()

# Plot 3: Baseline vs Decision Tree Comparison
fig, ax = plt.subplots(figsize=(7, 4))
models = ['Dummy Baseline\n(Most Frequent)', 'Decision Tree\n(max_depth=3)']
accuracies = [baseline_acc * 100, test_accuracy * 100]
bars = ax.bar(models, accuracies, color=['#7f7f7f', '#2ca02c'], width=0.45)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.set_title("Test Accuracy: Baseline vs. Decision Tree", fontsize=12, fontweight="bold")
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
comp_plot_path = "model_comparison.png"
plt.savefig(comp_plot_path, dpi=150)
print(f"[Saved] Model comparison plot saved to '{comp_plot_path}'.")
plt.close()

print("\nPipeline execution complete! You can open the saved plots or view them in VS Code.")
