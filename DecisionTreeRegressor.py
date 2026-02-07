import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_auc_score
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("F2BCMS.csv")

# Show first few rows
print(df.head())
print(f"Dataset shape: {df.shape}")

# Data Preparation
df["logQ2"] = np.log(df["Q^2"])  # Log-transform Q^2
df["logx"] = np.log(df["x"]) 
# Create feature matrix with original features
X = df[["logx", "logQ2"]]  # Use original x and log-transformed Q²
y = df["F2_exp"]  # Target variable

# Split the data FIRST to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Scale features (Decision Trees don't require scaling but we'll do it for consistency)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# For comparison, also scale the entire dataset for final plots
X_scaled_all = scaler_X.transform(X)

print("\n=== Training Decision Tree Model ===")

# Method 1: Basic Decision Tree with default parameters
print("Training basic Decision Tree model...")
dt_basic = DecisionTreeRegressor(random_state=42)
dt_basic.fit(X_train_scaled, y_train)

# Method 2: Decision Tree with hyperparameter tuning
print("\nTraining tuned Decision Tree model...")
dt_model = DecisionTreeRegressor(
    max_depth=5,           # Control tree depth to prevent overfitting
    min_samples_split=5,   # Minimum samples required to split a node
    min_samples_leaf=2,    # Minimum samples required at a leaf node
    max_features=None,     # Consider all features
    random_state=42
)
dt_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = dt_model.predict(X_train_scaled)
y_pred_test = dt_model.predict(X_test_scaled)
y_pred_all = dt_model.predict(X_scaled_all)

# ================================
# 1. REGRESSION METRICS
# ================================
print("\n" + "="*60)
print("REGRESSION METRICS")
print("="*60)

# Basic regression metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Additional regression metrics
from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, max_error

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
train_explained_var = explained_variance_score(y_train, y_pred_train)
test_explained_var = explained_variance_score(y_test, y_pred_test)
train_max_err = max_error(y_train, y_pred_train)
test_max_err = max_error(y_test, y_pred_test)

print("\n=== Model Evaluation ===")
print(f"Training MAE: {train_mae:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Training MSE: {train_mse:.6f}")
print(f"Test MSE: {test_mse:.6f}")
print(f"Training RMSE: {train_rmse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Training R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")
print(f"Training MAPE: {train_mape:.2f}%")
print(f"Test MAPE: {test_mape:.2f}%")
print(f"Training Explained Variance: {train_explained_var:.6f}")
print(f"Test Explained Variance: {test_explained_var:.6f}")
print(f"Training Max Error: {train_max_err:.6f}")
print(f"Test Max Error: {test_max_err:.6f}")

# ================================
# 2. CLASSIFICATION METRICS (Converted Problem)
# ================================
print("\n" + "="*60)
print("CLASSIFICATION METRICS (After Discretization)")
print("="*60)

# Convert regression to classification by binning F2 values
num_bins = 5

# Method 1: Equal-width bins
y_train_binned = pd.cut(y_train, bins=num_bins, labels=False)
y_test_binned = pd.cut(y_test, bins=num_bins, labels=False)
y_pred_train_binned = pd.cut(y_pred_train, bins=num_bins, labels=False)
y_pred_test_binned = pd.cut(y_pred_test, bins=num_bins, labels=False)

# Method 2: Quantile-based bins (for balanced classes)
y_train_binned_quantile = pd.qcut(y_train, q=num_bins, labels=False)
y_test_binned_quantile = pd.qcut(y_test, q=num_bins, labels=False)
y_pred_train_binned_quantile = pd.qcut(y_pred_train, q=num_bins, labels=False)
y_pred_test_binned_quantile = pd.qcut(y_pred_test, q=num_bins, labels=False)

# Calculate classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# For equal-width bins
print("\n--- Equal-width Binning ---")
accuracy = accuracy_score(y_test_binned, y_pred_test_binned)
precision = precision_score(y_test_binned, y_pred_test_binned, average='weighted', zero_division=0)
recall = recall_score(y_test_binned, y_pred_test_binned, average='weighted', zero_division=0)
f1 = f1_score(y_test_binned, y_pred_test_binned, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-Score (weighted): {f1:.4f}")

# For quantile-based bins
print("\n--- Quantile-based Binning ---")
accuracy_q = accuracy_score(y_test_binned_quantile, y_pred_test_binned_quantile)
precision_q = precision_score(y_test_binned_quantile, y_pred_test_binned_quantile, average='weighted', zero_division=0)
recall_q = recall_score(y_test_binned_quantile, y_pred_test_binned_quantile, average='weighted', zero_division=0)
f1_q = f1_score(y_test_binned_quantile, y_pred_test_binned_quantile, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy_q:.4f}")
print(f"Precision (weighted): {precision_q:.4f}")
print(f"Recall (weighted): {recall_q:.4f}")
print(f"F1-Score (weighted): {f1_q:.4f}")

# Classification report
print("\n--- Detailed Classification Report (Quantile Bins) ---")
print(classification_report(y_test_binned_quantile, y_pred_test_binned_quantile, zero_division=0))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_binned_quantile, y_pred_test_binned_quantile)
print("\n--- Confusion Matrix (Quantile Bins) ---")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Decision Tree)')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(range(num_bins))
plt.yticks(range(num_bins))
for i in range(num_bins):
    for j in range(num_bins):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', 
                 color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
plt.tight_layout()
plt.savefig('confusion matrix.pdf', format='pdf', bbox_inches='tight')
plt.show()
# ================================
# 3. THRESHOLD-BASED CLASSIFICATION
# ================================
print("\n" + "="*60)
print("THRESHOLD-BASED CLASSIFICATION")
print("="*60)

# Method 3: Binary classification based on threshold
# Let's classify F2 as "High" or "Low" based on median
median_f2 = np.median(y)
y_train_binary = (y_train > median_f2).astype(int)
y_test_binary = (y_test > median_f2).astype(int)
y_pred_train_binary = (y_pred_train > median_f2).astype(int)
y_pred_test_binary = (y_pred_test > median_f2).astype(int)

accuracy_binary = accuracy_score(y_test_binary, y_pred_test_binary)
precision_binary = precision_score(y_test_binary, y_pred_test_binary, zero_division=0)
recall_binary = recall_score(y_test_binary, y_pred_test_binary, zero_division=0)
f1_binary = f1_score(y_test_binary, y_pred_test_binary, zero_division=0)

print(f"Threshold: Median F2 = {median_f2:.4f}")
print(f"Accuracy: {accuracy_binary:.4f}")
print(f"Precision: {precision_binary:.4f}")
print(f"Recall: {recall_binary:.4f}")
print(f"F1-Score: {f1_binary:.4f}")

# Classification report for binary
print("\n--- Binary Classification Report ---")
print(classification_report(y_test_binary, y_pred_test_binary, 
                            target_names=['Low F2', 'High F2'], zero_division=0))

# Confusion matrix for binary
conf_matrix_binary = confusion_matrix(y_test_binary, y_pred_test_binary)
print("\n--- Binary Confusion Matrix ---")
print(conf_matrix_binary)

# Plot binary confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix_binary, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Decision Tree)')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Low F2', 'High F2'])
plt.yticks([0, 1], ['Low F2', 'High F2'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix_binary[i, j]), ha='center', va='center', 
                 color='white' if conf_matrix_binary[i, j] > conf_matrix_binary.max()/2 else 'black')
plt.tight_layout()
plt.savefig('f22.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Decision Tree Model Summary
print("\n=== Decision Tree Model Summary ===")
print(f"Max Depth: {dt_model.max_depth}")
print(f"Min Samples Split: {dt_model.min_samples_split}")
print(f"Min Samples Leaf: {dt_model.min_samples_leaf}")
print(f"Number of Features: {dt_model.n_features_in_}")
print(f"Number of Leaves: {dt_model.get_n_leaves()}")
print(f"Tree Depth: {dt_model.get_depth()}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Feature Importance')
plt.title('Decision Tree Feature Importance(Decision Tree)')
plt.tight_layout()
plt.savefig('f23.pdf', format='pdf', bbox_inches='tight')
plt.show()
# Plot 1: Actual vs Predicted (Test Set)
plt.figure(figsize=(10, 8))

# Test set predictions
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual F2')
plt.ylabel('Predicted F2')
plt.title('Test Set: Actual vs Predicted F2(Decision Tree)')
plt.grid(True, alpha=0.3)

# Training set predictions
plt.subplot(2, 2, 2)
plt.scatter(y_train, y_pred_train, alpha=0.6, color='green')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Actual F2')
plt.ylabel('Predicted F2')
plt.title('Training Set: Actual vs Predicted F2(Decision Tree)')
plt.grid(True, alpha=0.3)

# Residuals plot
plt.subplot(2, 2, 3)
residuals_test = y_test - y_pred_test
plt.scatter(y_pred_test, residuals_test, alpha=0.6, color='red')
plt.axhline(y=0, color='k', linestyle='--', linewidth=2)
plt.xlabel('Predicted F2')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Test Set: Residuals Plot(Decision Tree)')
plt.grid(True, alpha=0.3)

# Distribution of residuals
plt.subplot(2, 2, 4)
plt.hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals (Test Set)(Decision Tree)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('f24.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot 2: F2 vs x for first 20 points
plt.figure(figsize=(10, 6))

# Get first 20 points
first_20_indices = df.index[:20]
x_first_20 = df.loc[first_20_indices, 'x'].values
F2_actual_first_20 = df.loc[first_20_indices, 'F2_exp'].values

# Prepare and scale features for first 20 points
X_first_20 = X.loc[first_20_indices]
X_first_20_scaled = scaler_X.transform(X_first_20)
F2_pred_first_20 = dt_model.predict(X_first_20_scaled)

plt.scatter(x_first_20, F2_actual_first_20, c='blue', s=100, label='Actual F2', alpha=0.8)
plt.scatter(x_first_20, F2_pred_first_20, c='red', s=100, marker='s', label='Predicted F2', alpha=0.8)

# Connect actual and predicted with lines
for i in range(len(x_first_20)):
    plt.plot([x_first_20[i], x_first_20[i]], 
             [F2_actual_first_20[i], F2_pred_first_20[i]], 
             'k--', alpha=0.3, linewidth=0.5)

plt.xscale('log')
plt.xlabel('x (log scale)')
plt.ylabel('F2')
plt.title('First 20 Points: Actual vs Predicted F2 (Decision Tree)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('f24.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot 3: All data - Actual vs Predicted F2 vs x
plt.figure(figsize=(12, 8))

x_all_values = df["x"].values

plt.scatter(x_all_values, y, c='blue', alpha=0.6, s=30, label='Actual F2')
plt.scatter(x_all_values, y_pred_all, c='red', alpha=0.6, s=30, label='Predicted F2 (DT)')

plt.xscale('log')
plt.xlim(0.01, 1)
plt.xlabel('x (log scale)')
plt.ylabel('F2')
plt.title('All Data: Actual vs Predicted F2 vs x (Decision Tree)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('dt_scatter_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot 4: Error analysis by x and Q²
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error vs x (test set)
test_indices = y_test.index
x_test = df.loc[test_indices, 'x'].values
Q2_test = df.loc[test_indices, 'Q^2'].values
errors = np.abs(y_test.values - y_pred_test)

# Absolute error vs x
axes[0, 0].scatter(x_test, errors, alpha=0.6)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel('x (log scale)')
axes[0, 0].set_ylabel('Absolute Error')
axes[0, 0].set_title('Absolute Error vs x (Test Set)(Decision Tree)')
axes[0, 0].grid(True, alpha=0.3)

# Absolute error vs Q²
axes[0, 1].scatter(Q2_test, errors, alpha=0.6)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('Q² (log scale)')
axes[0, 1].set_ylabel('Absolute Error')
axes[0, 1].set_title('Absolute Error vs Q² (Test Set)(Decision Tree)')
axes[0, 1].grid(True, alpha=0.3)

# Relative error vs x
relative_errors = errors / (y_test.values + 1e-10)
axes[1, 0].scatter(x_test, relative_errors * 100, alpha=0.6)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('x (log scale)')
axes[1, 0].set_ylabel('Relative Error (%)')
axes[1, 0].set_title('Relative Error vs x (Test Set)(Decision Tree)')
axes[1, 0].grid(True, alpha=0.3)

# Distribution of errors
axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Absolute Errors (Test Set(Decision Tree))')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot 5: Decision Tree visualization (small version for readability)
plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          feature_names=X.columns, 
          filled=True, 
          rounded=True,
          max_depth=3,  # Show only first 3 levels for readability
          fontsize=10)
plt.title('Decision Tree Visualization (First 3 levels)(Decision Tree)')
plt.tight_layout()
plt.savefig('f25.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot 6: Effect of tree depth on performance
depths = range(1, 21)
train_scores = []
test_scores = []

for depth in depths:
    dt_temp = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt_temp.fit(X_train_scaled, y_train)
    train_scores.append(r2_score(y_train, dt_temp.predict(X_train_scaled)))
    test_scores.append(r2_score(y_test, dt_temp.predict(X_test_scaled)))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training R²')
plt.plot(depths, test_scores, 's-', label='Test R²')
plt.xlabel('Tree Depth')
plt.ylabel('R² Score')
plt.title('Effect of Tree Depth on Model Performance(Decision Tree)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('f26.pdf', format='pdf', bbox_inches='tight')
plt.show()
# Cross-validation for Decision Tree
print("\n=== Performing Cross-Validation ===")
cv_scores = cross_val_score(dt_model, X_scaled_all, y, cv=5, scoring='r2')
print(f"Cross-Validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.6f}")
print(f"CV R² std: {cv_scores.std():.6f}")

# Make predictions for specific points
def predict_new_point(x_value, Q2_value):
    # Create feature array
    logQ2 = np.log(Q2_value)
    new_features = np.array([[x_value[0], logQ2[0]]])
    
    # Scale features using the same scaler
    new_features_scaled = scaler_X.transform(new_features)
    
    # Predict
    prediction = dt_model.predict(new_features_scaled)
    
    print(f"\n=== New Prediction (Decision Tree) ===")
    print(f"Input: x = {x_value[0]:.6f}, Q² = {Q2_value[0]:.6f}")
    print(f"Predicted F2: {prediction[0]:.6f}")
    
    return prediction[0]

# Example predictions
print("\n=== Example Predictions (Decision Tree) ===")

# First sample
sample_1_x = df["x"].iloc[0:1].values
sample_1_Q2 = df["Q^2"].iloc[0:1].values
pred_1 = predict_new_point(sample_1_x, sample_1_Q2)
print(f"Actual F2: {df['F2_exp'].iloc[0]:.6f}")

# 500th sample
sample_500_x = df["x"].iloc[500:501].values
sample_500_Q2 = df["Q^2"].iloc[500:501].values
pred_500 = predict_new_point(sample_500_x, sample_500_Q2)
print(f"Actual F2: {df['F2_exp'].iloc[500]:.6f}")

# Custom point
custom_x = np.array([0.1])
custom_Q2 = np.array([10.0])
predict_new_point(custom_x, custom_Q2)

# Save results to CSV
results_df = pd.DataFrame({
    'x': df['x'],
    'Q^2': df['Q^2'],
    'logQ2': df['logQ2'],
    'F2_actual': df['F2_exp'],
    'F2_predicted': y_pred_all,
    'error': df['F2_exp'] - y_pred_all,
    'abs_error': np.abs(df['F2_exp'] - y_pred_all),
    'relative_error': (df['F2_exp'] - y_pred_all) / (df['F2_exp'] + 1e-10),
    'F2_category': pd.qcut(df['F2_exp'], q=num_bins, labels=False),
    'F2_binary': (df['F2_exp'] > np.median(df['F2_exp'])).astype(int)
})

results_df.to_csv('decision_tree_predictions.csv', index=False)
print("\nPredictions saved to 'decision_tree_predictions.csv'")

# Final comparison
print("\n" + "="*60)
print("DECISION TREE MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"\nREGRESSION METRICS:")
print(f"  Test R² Score: {test_r2:.6f}")
print(f"  Test MAE: {test_mae:.6f}")
print(f"  Test RMSE: {test_rmse:.6f}")
print(f"  Test MAPE: {test_mape:.2f}%")

print(f"\nCLASSIFICATION METRICS (Quantile Bins):")
print(f"  Accuracy: {accuracy_q:.4f}")
print(f"  Precision (weighted): {precision_q:.4f}")
print(f"  Recall (weighted): {recall_q:.4f}")
print(f"  F1-Score (weighted): {f1_q:.4f}")

print(f"\nCLASSIFICATION METRICS (Binary - Above/Below Median):")
print(f"  Accuracy: {accuracy_binary:.4f}")
print(f"  Precision: {precision_binary:.4f}")
print(f"  Recall: {recall_binary:.4f}")
print(f"  F1-Score: {f1_binary:.4f}")

print(f"\nMODEL DETAILS:")
print(f"  Tree Depth: {dt_model.get_depth()}")
print(f"  Number of Leaves: {dt_model.get_n_leaves()}")
print(f"  Most Important Feature: {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['importance']:.4f})")
print("="*60)

# Additional: Check for overfitting
print("\n=== Overfitting Analysis ===")
print(f"Training R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")
print(f"Difference (Training - Test): {train_r2 - test_r2:.6f}")

if train_r2 - test_r2 > 0.2:
    print("Warning: Model may be overfitting (large gap between training and test R²)")
elif train_r2 - test_r2 > 0.1:
    print("Note: Moderate gap between training and test R²")
else:
    print("Good: Small gap between training and test R²")

    # ======================================================
# FINAL SUMMARY TABLES (SAFE VERSION)
# ======================================================

import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

print(">>> STARTING SUMMARY TABLES <<<")

# =========================
# BINARY CLASSIFICATION METRICS
# =========================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Metrics from Regressor (threshold-based)
# -------------------------
accuracy  = accuracy_score(y_test_binary, y_pred_test_binary)
precision = precision_score(y_test_binary, y_pred_test_binary)
recall    = recall_score(y_test_binary, y_pred_test_binary)
f1        = f1_score(y_test_binary, y_pred_test_binary)

# -------------------------
# ROC-AUC using Classifier (CORRECT WAY)
# -------------------------
dt_classifier = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

dt_classifier.fit(X_train_scaled, y_train_binary)

y_test_proba = dt_classifier.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test_binary, y_test_proba)

print(f"ROC-AUC Score (Binary): {roc_auc:.4f}")

# -------------------------
# Summary Table
# -------------------------
classification_summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Value":  [accuracy, precision, recall, f1, roc_auc]
})

print("\nClassification Metrics Summary:")
print(classification_summary)

# -------------------------
# Save to Excel
# -------------------------
classification_summary.to_excel(
    "classification_metrics_decision_tree.xlsx",
    index=False
)

# -------------------------
# Save to PDF
# -------------------------
with PdfPages("classification_metrics_decision_tree.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    table = ax.table(
        cellText=classification_summary.values,
        colLabels=classification_summary.columns,
        cellLoc='center',
        loc='center'
    )

    table.scale(1, 1.5)
    ax.set_title("Classification Metrics – Decision Tree", fontsize=12)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# =========================
# REGRESSION METRICS
# =========================

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae  = mean_absolute_error(y_test, y_pred_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse  = mean_squared_error(y_test, y_pred_test)

train_rmse = np.sqrt(train_mse)
test_rmse  = np.sqrt(test_mse)

train_r2 = r2_score(y_train, y_pred_train)
test_r2  = r2_score(y_test, y_pred_test)

regression_summary = pd.DataFrame({
    "Metric": [
        "Train MAE", "Test MAE",
        "Train MSE", "Test MSE",
        "Train RMSE", "Test RMSE",
        "Train R2", "Test R2"
    ],
    "Value": [
        train_mae, test_mae,
        train_mse, test_mse,
        train_rmse, test_rmse,
        train_r2, test_r2
    ]
})

print(regression_summary)

regression_summary.to_excel(
    "regression_metrics_decision_tree.xlsx",
    index=False
)

with PdfPages("regression_metrics_decision_tree.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    ax.table(
        cellText=regression_summary.values,
        colLabels=regression_summary.columns,
        cellLoc='center',
        loc='center'
    )
    ax.set_title("Regression Metrics (Decision Tree)")
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print("✅ SUMMARY TABLES SAVED SUCCESSFULLY")
print("📂 Directory:", os.getcwd())

