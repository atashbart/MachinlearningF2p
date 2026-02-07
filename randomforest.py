import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import matplotlib
warnings.filterwarnings('ignore')

# =====================================================
# Load data
# =====================================================
df = pd.read_csv("F2BCMS.csv")
print(df.head())
print(f"Dataset shape: {df.shape}")

# =====================================================
# Data preparation
# =====================================================
df["logQ2"] = np.log(df["Q^2"])
df["logx"] = np.log(df["x"].clip(lower=1e-6))

X = df[["logx", "logQ2"]]
y = df["F2_exp"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Scaling (kept for identical structure)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
X_scaled_all = scaler_X.transform(X)

# =====================================================
# APPROACH 1: RANDOM FOREST REGRESSION
# =====================================================
print("\n" + "="*60)
print("APPROACH 1: RANDOM FOREST REGRESSION")
print("="*60)

rf_reg_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_reg_model.fit(X_train_scaled, y_train)

y_pred_train_reg = rf_reg_model.predict(X_train_scaled)
y_pred_test_reg = rf_reg_model.predict(X_test_scaled)
y_pred_all_reg = rf_reg_model.predict(X_scaled_all)

# =====================================================
# REGRESSION METRICS
# =====================================================
print("\n" + "="*60)
print("REGRESSION METRICS")
print("="*60)

train_mae = mean_absolute_error(y_train, y_pred_train_reg)
test_mae = mean_absolute_error(y_test, y_pred_test_reg)
train_mse = mean_squared_error(y_train, y_pred_train_reg)
test_mse = mean_squared_error(y_test, y_pred_test_reg)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_pred_train_reg)
test_r2 = r2_score(y_test, y_pred_test_reg)

from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, max_error

print(f"Training MAE: {train_mae:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Training RMSE: {train_rmse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Training R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")

# =====================================================
# APPROACH 2: RANDOM FOREST CLASSIFICATION (Multi-class)
# =====================================================
print("\n" + "="*60)
print("APPROACH 2: RANDOM FOREST CLASSIFICATION")
print("="*60)

num_bins = 5
y_train_class = pd.qcut(y_train, q=num_bins, labels=False)
y_test_class = pd.qcut(y_test, q=num_bins, labels=False)
bin_edges = pd.qcut(y_train, q=num_bins, retbins=True)[1]

rf_clf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_clf_model.fit(X_train_scaled, y_train_class)

y_pred_test_class = rf_clf_model.predict(X_test_scaled)
y_pred_proba_test = rf_clf_model.predict_proba(X_test_scaled)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test_class, y_pred_test_class)
precision = precision_score(y_test_class, y_pred_test_class, average='weighted')
recall = recall_score(y_test_class, y_pred_test_class, average='weighted')
f1 = f1_score(y_test_class, y_pred_test_class, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# =====================================================
# APPROACH 3: BINARY CLASSIFICATION
# =====================================================
print("\n" + "="*60)
print("APPROACH 3: BINARY CLASSIFICATION")
print("="*60)

median_f2 = np.median(y)
y_train_binary = (y_train > median_f2).astype(int)
y_test_binary = (y_test > median_f2).astype(int)

rf_binary = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_binary.fit(X_train_scaled, y_train_binary)

y_pred_test_binary = rf_binary.predict(X_test_scaled)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test_binary, y_pred_test_binary))

# =====================================================
# DECISION BOUNDARY (Binary)
# =====================================================
plt.figure(figsize=(10, 6))

x_min, x_max = X_test_scaled[:, 0].min()-0.5, X_test_scaled[:, 0].max()+0.5
y_min, y_max = X_test_scaled[:, 1].min()-0.5, X_test_scaled[:, 1].max()+0.5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = rf_binary.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# =====================================================
# SAVE RESULTS
# =====================================================
results_df = pd.DataFrame({
    "x": df["x"],
    "Q^2": df["Q^2"],
    "F2_actual": y,
    "F2_predicted_RF": y_pred_all_reg,
    "error": y - y_pred_all_reg,
    "abs_error": np.abs(y - y_pred_all_reg)
})
from sklearn.metrics import confusion_matrix

conf_matrix_rf = confusion_matrix(y_test_class, y_pred_test_class)

print("\n--- Confusion Matrix (Random Forest - Multi-class) ---")
print(conf_matrix_rf)


plt.imshow(conf_matrix_rf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Random Forest)')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(range(num_bins))
plt.yticks(range(num_bins))

for i in range(num_bins):
    for j in range(num_bins):
        plt.text(j, i, conf_matrix_rf[i, j],
                 ha='center', va='center',
                 color='white' if conf_matrix_rf[i, j] > conf_matrix_rf.max()/2 else 'black')

plt.tight_layout()
plt.savefig('RF_confusion_matrix_multiclass.pdf', format='pdf', bbox_inches='tight')
plt.show()
conf_matrix_binary_rf = confusion_matrix(y_test_binary, y_pred_test_binary)

print("\n--- Binary Confusion Matrix (Random Forest) ---")
print(conf_matrix_binary_rf)

plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix_binary_rf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Random Forest Binary)')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Low F2', 'High F2'])
plt.yticks([0, 1], ['Low F2', 'High F2'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix_binary_rf[i, j],
                 ha='center', va='center',
                 color='white' if conf_matrix_binary_rf[i, j] > conf_matrix_binary_rf.max()/2 else 'black')

plt.tight_layout()
plt.savefig('RF_confusion_matrix_binary.pdf', format='pdf', bbox_inches='tight')
plt.show()
print("\n=== RANDOM FOREST MODEL SUMMARY ===")
print(f"Number of trees (n_estimators): {rf_reg_model.n_estimators}")
print(f"Max depth: {rf_reg_model.max_depth}")
print(f"Min samples split: {rf_reg_model.min_samples_split}")
print(f"Min samples leaf: {rf_reg_model.min_samples_leaf}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Test
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_test_reg, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual F2')
plt.ylabel('Predicted F2')
plt.title('Test: Actual vs Predicted F2 (Random Forest)')
plt.grid(True, alpha=0.3)

# Train
plt.subplot(2, 2, 2)
plt.scatter(y_train, y_pred_train_reg, alpha=0.6, color='green')
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()], 'k--')
plt.xlabel('Actual F2')
plt.ylabel('Predicted F2')
plt.title('Train: Actual vs Predicted F2 (Random Forest)')
plt.grid(True, alpha=0.3)

# Residuals
plt.subplot(2, 2, 3)
residuals_rf = y_test - y_pred_test_reg
plt.scatter(y_pred_test_reg, residuals_rf, alpha=0.6, color='red')
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('Predicted F2')
plt.ylabel('Residuals')
plt.title('Residuals (Test Set)(Random Forest)')
plt.grid(True, alpha=0.3)

# Residual distribution
plt.subplot(2, 2, 4)
plt.hist(residuals_rf, bins=30, edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution (Test Set)(Random Forest)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RF_actual_vs_predicted.pdf', format='pdf', bbox_inches='tight')
plt.show()
n_estimators_range = range(10, 310, 10)
train_r2_rf = []
test_r2_rf = []

for n in n_estimators_range:
    rf_tmp = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf_tmp.fit(X_train_scaled, y_train)
    train_r2_rf.append(r2_score(y_train, rf_tmp.predict(X_train_scaled)))
    test_r2_rf.append(r2_score(y_test, rf_tmp.predict(X_test_scaled)))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_r2_rf, 'o-', label='Training R²')
plt.plot(n_estimators_range, test_r2_rf, 's-', label='Test R²')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('R² Score')
plt.title('Performance vs n_estimators (Random Forest)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('RF_performance_vs_estimators.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.figure(figsize=(12, 5))

x_min, x_max = X_test_scaled[:, 0].min()-0.5, X_test_scaled[:, 0].max()+0.5
y_min, y_max = X_test_scaled[:, 1].min()-0.5, X_test_scaled[:, 1].max()+0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = rf_binary.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
            c=y_test_binary, edgecolors='k', cmap=plt.cm.RdBu)
plt.title('Random Forest Decision Boundary (Binary)(Random Forest)')
plt.xlabel('logx (scaled)')
plt.ylabel('logQ2 (scaled)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
            c=y_train_binary, cmap=plt.cm.RdBu, alpha=0.6)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
            c=y_test_binary, cmap=plt.cm.RdBu, marker='s', edgecolors='k')
plt.title('Train/Test Samples (Binary)(Random Forest)')
plt.xlabel('logx (scaled)')
plt.ylabel('logQ2 (scaled)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RF_decision_boundary.pdf', format='pdf', bbox_inches='tight')
plt.show()
results_df.to_csv("random_forest_predictions.csv", index=False)
plt.figure(figsize=(10, 6))

sc = plt.scatter(
    results_df["x"],
    results_df["F2_actual"],
    c=results_df["Q^2"],
    cmap='viridis',
    alpha=0.7,
    label='Actual F2'
)

plt.scatter(
    results_df["x"],
    results_df["F2_predicted_RF"],
    c=results_df["Q^2"],
    cmap='viridis',
    marker='x',
    alpha=0.7,
    label='Predicted F2 (RF)'
)

plt.colorbar(sc, label=r'$Q^2$')
plt.xlabel('x')
plt.ylabel('F2')
plt.title('All Data: Actual vs Predicted F2 vs x (Random Forest)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    'RF_all_data_actual_vs_predicted_F2_vs_x_Q2.pdf',
    format='pdf',
    bbox_inches='tight'
)
plt.show()
plt.close()

print("\nPredictions saved to 'random_forest_predictions.csv'")
# =====================================================
# FINAL TABLE 1: CLASSIFICATION METRICS (RANDOM FOREST)
# =====================================================

from sklearn.metrics import roc_auc_score

# ROC-AUC (multi-class, One-vs-Rest)
try:
    roc_auc_multiclass = roc_auc_score(
        y_test_class,
        y_pred_proba_test,
        multi_class='ovr',
        average='weighted'
    )
except:
    roc_auc_multiclass = np.nan

classification_metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision (weighted)",
        "Recall (weighted)",
        "F1-score (weighted)",
        "ROC-AUC (OvR)"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc_multiclass
    ]
})

print("\n=== CLASSIFICATION METRICS TABLE (Random Forest) ===")
print(classification_metrics_df)

# -------- Save to Excel --------
classification_metrics_df.to_excel(
    "RF_classification_metrics.xlsx",
    index=False
)

# -------- Save to PDF --------
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')
ax.axis('tight')

table = ax.table(
    cellText=classification_metrics_df.round(4).values,
    colLabels=classification_metrics_df.columns,
    loc='center'
)
table.scale(1, 1.6)

plt.title("Classification Metrics (Random Forest)", fontsize=12)
plt.savefig(
    "RF_classification_metrics.pdf",
    bbox_inches='tight'
)
plt.close()
# =====================================================
# FINAL TABLE 2: REGRESSION METRICS (RANDOM FOREST)
# =====================================================

regression_metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],
    "Training": [
        train_mae,
        train_mse,
        train_rmse,
        train_r2
    ],
    "Test": [
        test_mae,
        test_mse,
        test_rmse,
        test_r2
    ]
})

print("\n=== REGRESSION METRICS TABLE (Random Forest) ===")
print(regression_metrics_df)

# -------- Save to Excel --------
regression_metrics_df.to_excel(
    "RF_regression_metrics.xlsx",
    index=False
)

# -------- Save to PDF --------
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')
ax.axis('tight')

table = ax.table(
    cellText=regression_metrics_df.round(6).values,
    colLabels=regression_metrics_df.columns,
    loc='center'
)
table.scale(1, 1.6)

plt.title("Regression Metrics (Random Forest)", fontsize=12)
plt.savefig(
    "RF_regression_metrics.pdf",
    bbox_inches='tight'
)
plt.close()

