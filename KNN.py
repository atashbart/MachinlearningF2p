import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("F2BCMS.csv")

# Show first few rows
print(df.head())
print(f"Dataset shape: {df.shape}")

# Data Preparation
df["logQ2"] = np.log(df["Q^2"])  # Log-transform Q^2
df["logx"] = np.log(df["x"].clip(lower=1e-6))
# Create feature matrix with original features
X = df[["logx", "logQ2"]]  # Use original x and log-transformed Q²
y = df["F2_exp"]  # Target variable (continuous)

# Split the data FIRST to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Scale features (KNN requires feature scaling due to distance-based algorithm)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# For comparison, also scale the entire dataset for final plots
X_scaled_all = scaler_X.transform(X)

# ================================
# APPROACH 1: KNN REGRESSION
# ================================
print("\n" + "="*60)
print("APPROACH 1: KNN REGRESSION")
print("="*60)

print("\n=== Training KNN Regression Model ===")

# Method 1: Basic KNN Regression with default parameters
print("Training basic KNN Regression model...")
knn_reg_basic = KNeighborsRegressor()
knn_reg_basic.fit(X_train_scaled, y_train)

# Method 2: Tuned KNN Regression
print("\nTraining tuned KNN Regression model...")
knn_reg_model = KNeighborsRegressor(
    n_neighbors=5,        # Number of neighbors to consider
    weights='distance',   # Weight points by inverse of distance
    algorithm='auto',     # Algorithm to compute nearest neighbors
    leaf_size=30,         # Leaf size for KDTree/BallTree
    p=2,                  # Power parameter (2 = Euclidean distance)
    metric='minkowski'    # Distance metric
)
knn_reg_model.fit(X_train_scaled, y_train)

# Predictions for regression
y_pred_train_reg = knn_reg_model.predict(X_train_scaled)
y_pred_test_reg = knn_reg_model.predict(X_test_scaled)
y_pred_all_reg = knn_reg_model.predict(X_scaled_all)

# ================================
# 1. REGRESSION METRICS
# ================================
print("\n" + "="*60)
print("REGRESSION METRICS")
print("="*60)

# Basic regression metrics
train_mae = mean_absolute_error(y_train, y_pred_train_reg)
test_mae = mean_absolute_error(y_test, y_pred_test_reg)
train_mse = mean_squared_error(y_train, y_pred_train_reg)
test_mse = mean_squared_error(y_test, y_pred_test_reg)
train_r2 = r2_score(y_train, y_pred_train_reg)
test_r2 = r2_score(y_test, y_pred_test_reg)

# Additional regression metrics
from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, max_error

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mape = mean_absolute_percentage_error(y_train, y_pred_train_reg) * 100
test_mape = mean_absolute_percentage_error(y_test, y_pred_test_reg) * 100
train_explained_var = explained_variance_score(y_train, y_pred_train_reg)
test_explained_var = explained_variance_score(y_test, y_pred_test_reg)
train_max_err = max_error(y_train, y_pred_train_reg)
test_max_err = max_error(y_test, y_pred_test_reg)

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
# APPROACH 2: KNN CLASSIFICATION
# ================================
print("\n" + "="*60)
print("APPROACH 2: KNN CLASSIFICATION")
print("="*60)

# Convert continuous F2 values into classes
num_bins = 5
# Method 1: Quantile-based bins (for balanced classes)
y_train_class = pd.qcut(y_train, q=num_bins, labels=False)
y_test_class = pd.qcut(y_test, q=num_bins, labels=False)

# Get bin edges for later use
bin_edges = pd.qcut(y_train, q=num_bins, retbins=True)[1]

print(f"\nClass distribution in training set:")
print(pd.Series(y_train_class).value_counts().sort_index())
print(f"\nClass distribution in test set:")
print(pd.Series(y_test_class).value_counts().sort_index())

print("\n=== Training KNN Classification Model ===")

# Tuned KNN Classification
print("\nTraining tuned KNN Classification model...")
knn_clf_model = KNeighborsClassifier(
    n_neighbors=5,        # Number of neighbors to consider
    weights='distance',   # Weight points by inverse of distance
    algorithm='auto',     # Algorithm to compute nearest neighbors
    leaf_size=30,         # Leaf size for KDTree/BallTree
    p=2,                  # Power parameter (2 = Euclidean distance)
    metric='minkowski'    # Distance metric
)
knn_clf_model.fit(X_train_scaled, y_train_class)

# Predictions for classification
y_pred_train_class = knn_clf_model.predict(X_train_scaled)
y_pred_test_class = knn_clf_model.predict(X_test_scaled)
y_pred_proba_train = knn_clf_model.predict_proba(X_train_scaled)
y_pred_proba_test = knn_clf_model.predict_proba(X_test_scaled)

# ================================
# 2. CLASSIFICATION METRICS
# ================================
print("\n" + "="*60)
print("CLASSIFICATION METRICS")
print("="*60)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Calculate classification metrics
accuracy = accuracy_score(y_test_class, y_pred_test_class)
precision = precision_score(y_test_class, y_pred_test_class, average='weighted', zero_division=0)
recall = recall_score(y_test_class, y_pred_test_class, average='weighted', zero_division=0)
f1 = f1_score(y_test_class, y_pred_test_class, average='weighted', zero_division=0)

# For multi-class ROC-AUC (requires one-vs-rest or one-vs-one)
try:
    roc_auc = roc_auc_score(y_test_class, y_pred_proba_test, multi_class='ovr', average='weighted')
    roc_available = True
except:
    roc_available = False

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-Score (weighted): {f1:.4f}")
if roc_available:
    print(f"ROC-AUC Score (weighted, one-vs-rest): {roc_auc:.4f}")

# Classification report
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test_class, y_pred_test_class, zero_division=0))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_class, y_pred_test_class)
print("\n--- Confusion Matrix ---")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (KNN Classification)')
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
# 3. BINARY CLASSIFICATION
# ================================
print("\n" + "="*60)
print("APPROACH 3: BINARY CLASSIFICATION")
print("="*60)

# Convert to binary classification based on median
median_f2 = np.median(y)
y_train_binary = (y_train > median_f2).astype(int)
y_test_binary = (y_test > median_f2).astype(int)

print(f"Threshold: Median F2 = {median_f2:.4f}")
print(f"Class distribution (binary):")
print(f"  Class 0 (Low F2): {sum(y_test_binary == 0)} samples")
print(f"  Class 1 (High F2): {sum(y_test_binary == 1)} samples")

print("\n=== Training KNN Model (Binary) ===")
knn_binary = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski'
)
knn_binary.fit(X_train_scaled, y_train_binary)

# Predictions for binary
y_pred_train_binary = knn_binary.predict(X_train_scaled)
y_pred_test_binary = knn_binary.predict(X_test_scaled)
y_pred_proba_train_binary = knn_binary.predict_proba(X_train_scaled)
y_pred_proba_test_binary = knn_binary.predict_proba(X_test_scaled)

# Binary classification metrics
accuracy_binary = accuracy_score(y_test_binary, y_pred_test_binary)
precision_binary = precision_score(y_test_binary, y_pred_test_binary, zero_division=0)
recall_binary = recall_score(y_test_binary, y_pred_test_binary, zero_division=0)
f1_binary = f1_score(y_test_binary, y_pred_test_binary, zero_division=0)

try:
    roc_auc_binary = roc_auc_score(y_test_binary, y_pred_proba_test_binary[:, 1])
    roc_available_binary = True
except:
    roc_available_binary = False

print("\n=== Binary Model Evaluation ===")
print(f"Accuracy: {accuracy_binary:.4f}")
print(f"Precision: {precision_binary:.4f}")
print(f"Recall: {recall_binary:.4f}")
print(f"F1-Score: {f1_binary:.4f}")
if roc_available_binary:
    print(f"ROC-AUC Score: {roc_auc_binary:.4f}")

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
plt.title('Confusion Matrix (KNN Regression)')
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
plt.savefig('binary confusion matrix.pdf', format='pdf', bbox_inches='tight')
plt.show

# KNN Model Summary
print("\n=== KNN Model Summary ===")
print(f"Number of neighbors: {knn_reg_model.n_neighbors}")
print(f"Weights: {knn_reg_model.weights}")
print(f"Algorithm: {knn_reg_model.algorithm}")
print(f"Leaf size: {knn_reg_model.leaf_size}")
print(f"Distance metric: {knn_reg_model.metric}")
print(f"Power parameter (p): {knn_reg_model.p}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Plot 1: Actual vs Predicted (Regression)
plt.figure(figsize=(10, 8))

# Test set predictions
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_test_reg, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual F2')
plt.ylabel('Predicted F2')
plt.title('Test Set: Actual vs Predicted F2 (KNN Regression)')
plt.grid(True, alpha=0.3)

# Training set predictions
plt.subplot(2, 2, 2)
plt.scatter(y_train, y_pred_train_reg, alpha=0.6, color='green')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Actual F2')
plt.ylabel('Predicted F2')
plt.title('Training Set: Actual vs Predicted F2 (KNN Regression)')
plt.grid(True, alpha=0.3)

# Residuals plot
plt.subplot(2, 2, 3)
residuals_test = y_test - y_pred_test_reg
plt.scatter(y_pred_test_reg, residuals_test, alpha=0.6, color='red')
plt.axhline(y=0, color='k', linestyle='--', linewidth=2)
plt.xlabel('Predicted F2')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Test Set: Residuals Plot(KNN Regression)')
plt.grid(True, alpha=0.3)

# Distribution of residuals
plt.subplot(2, 2, 4)
plt.hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals (Test Set)(KNN Regression)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Actual vs Predicted.pdf', format='pdf', bbox_inches='tight')
plt.show

# Plot 2: Effect of k (number of neighbors) on performance
k_values = range(1, 31)
train_scores_reg = []
test_scores_reg = []
train_scores_clf = []
test_scores_clf = []

for k in k_values:
    # For regression
    knn_temp_reg = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn_temp_reg.fit(X_train_scaled, y_train)
    train_scores_reg.append(r2_score(y_train, knn_temp_reg.predict(X_train_scaled)))
    test_scores_reg.append(r2_score(y_test, knn_temp_reg.predict(X_test_scaled)))
    
    # For classification
    knn_temp_clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn_temp_clf.fit(X_train_scaled, y_train_class)
    train_scores_clf.append(accuracy_score(y_train_class, knn_temp_clf.predict(X_train_scaled)))
    test_scores_clf.append(accuracy_score(y_test_class, knn_temp_clf.predict(X_test_scaled)))

plt.figure(figsize=(14, 6))

# Regression performance vs k
plt.subplot(1, 2, 1)
plt.plot(k_values, train_scores_reg, 'o-', label='Training R²')
plt.plot(k_values, test_scores_reg, 's-', label='Test R²')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('R² Score')
plt.title('Performance vs Number of Neighbors(KNN Regression)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=5, color='r', linestyle='--', alpha=0.5, label='k=5 (selected)')
plt.legend()

# Classification performance vs k
plt.subplot(1, 2, 2)
plt.plot(k_values, train_scores_clf, 'o-', label='Training Accuracy')
plt.plot(k_values, test_scores_clf, 's-', label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Performance vs Number of Neighbors(KNN Regression)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('Regression performance vs k.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Plot 3: Decision boundary visualization (for binary classification)
plt.figure(figsize=(12, 5))

# Create a mesh grid for visualization
x_min, x_max = X_test_scaled[:, 0].min() - 0.5, X_test_scaled[:, 0].max() + 0.5
y_min, y_max = X_test_scaled[:, 1].min() - 0.5, X_test_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict for mesh grid
Z = knn_binary.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], 
            c=y_test_binary, edgecolors='k', cmap=plt.cm.RdBu, s=30)
plt.xlabel('x (scaled)')
plt.ylabel('logQ2 (scaled)')
plt.title('KNN Decision Boundary (Binary)(KNN Regression)')
plt.grid(True, alpha=0.3)

# Plot training points with their labels
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                      c=y_train_binary, cmap=plt.cm.RdBu, alpha=0.7, s=50)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], 
            c=y_test_binary, cmap=plt.cm.RdBu, alpha=0.7, s=100, marker='s', edgecolors='k')
plt.xlabel('x (scaled)')
plt.ylabel('logQ2 (scaled)')
plt.title('Training and Test Data with True Labels(KNN Regression)')
plt.legend(handles=scatter.legend_elements()[0], labels=['Low F2', 'High F2'])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Decision boundary visualization.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Plot 4: All data - Actual vs Predicted F2 vs x
plt.figure(figsize=(12, 8))

x_all_values = df["x"].values

plt.scatter(x_all_values, y, c='blue', alpha=0.6, s=30, label='Actual F2')
plt.scatter(x_all_values, y_pred_all_reg, c='red', alpha=0.6, s=30, label='Predicted F2 (KNN)')

plt.xscale('log')
plt.xlim(0.01, 1)
plt.xlabel('x (log scale)')
plt.ylabel('F2')
plt.title('All Data: Actual vs Predicted F2 vs x (KNN Regression)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('knn_scatter_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot 5: Error analysis by x and Q²
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error vs x (test set)
test_indices = y_test.index
x_test = df.loc[test_indices, 'x'].values
Q2_test = df.loc[test_indices, 'Q^2'].values
errors = np.abs(y_test.values - y_pred_test_reg)

# Absolute error vs x
axes[0, 0].scatter(x_test, errors, alpha=0.6)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel('x (log scale)')
axes[0, 0].set_ylabel('Absolute Error')
axes[0, 0].set_title('Absolute Error vs x (Test Set)(KNN Regression)')
axes[0, 0].grid(True, alpha=0.3)

# Absolute error vs Q²
axes[0, 1].scatter(Q2_test, errors, alpha=0.6)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('Q² (log scale)')
axes[0, 1].set_ylabel('Absolute Error')
axes[0, 1].set_title('Absolute Error vs Q² (Test Set)(KNN Regression)')
axes[0, 1].grid(True, alpha=0.3)

# Relative error vs x
relative_errors = errors / (y_test.values + 1e-10)
axes[1, 0].scatter(x_test, relative_errors * 100, alpha=0.6)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('x (log scale)')
axes[1, 0].set_ylabel('Relative Error (%)')
axes[1, 0].set_title('Relative Error vs x (Test Set)(KNN Regression)')
axes[1, 0].grid(True, alpha=0.3)

# Distribution of errors
axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Absolute Errors (Test Set)(KNN Regression)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(' Distribution of errors.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Plot 6: Effect of distance weighting
weights_options = ['uniform', 'distance']
uniform_scores = []
distance_scores = []

for weights in weights_options:
    knn_temp = KNeighborsRegressor(n_neighbors=5, weights=weights)
    knn_temp.fit(X_train_scaled, y_train)
    uniform_scores.append(r2_score(y_train, knn_temp.predict(X_train_scaled)))
    distance_scores.append(r2_score(y_test, knn_temp.predict(X_test_scaled)))

plt.figure(figsize=(8, 6))
x_pos = np.arange(len(weights_options))
width = 0.35

plt.bar(x_pos - width/2, uniform_scores, width, label='Training R²', alpha=0.8)
plt.bar(x_pos + width/2, distance_scores, width, label='Test R²', alpha=0.8)
plt.xlabel('Weighting Method')
plt.ylabel('R² Score')
plt.title('Effect of Distance Weighting on KNN Performance(KNN Regression)')
plt.xticks(x_pos, weights_options)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Effect of distance weighting.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Cross-validation for KNN Regression
print("\n=== Performing Cross-Validation (Regression) ===")
cv_scores_reg = cross_val_score(knn_reg_model, X_scaled_all, y, cv=5, scoring='r2')
print(f"Cross-Validation R² scores: {cv_scores_reg}")
print(f"Mean CV R²: {cv_scores_reg.mean():.6f}")
print(f"CV R² std: {cv_scores_reg.std():.6f}")

# Cross-validation for KNN Classification
print("\n=== Performing Cross-Validation (Classification) ===")
cv_scores_clf = cross_val_score(knn_clf_model, X_scaled_all, pd.qcut(y, q=num_bins, labels=False), 
                               cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy scores: {cv_scores_clf}")
print(f"Mean CV Accuracy: {cv_scores_clf.mean():.6f}")
print(f"CV Accuracy std: {cv_scores_clf.std():.6f}")

# Make predictions for specific points
def predict_new_point(x_value, Q2_value, regression=True):
    # Create feature array
    logQ2 = np.log(Q2_value)
    new_features = np.array([[x_value[0], logQ2[0]]])
    
    # Scale features using the same scaler
    new_features_scaled = scaler_X.transform(new_features)
    
    if regression:
        # Regression prediction
        prediction = knn_reg_model.predict(new_features_scaled)[0]
        
        print(f"\n=== New Prediction (KNN Regression) ===")
        print(f"Input: x = {x_value[0]:.6f}, Q² = {Q2_value[0]:.6f}")
        print(f"Predicted F2: {prediction:.6f}")
        
        return prediction
    else:
        # Classification prediction
        prediction_class = knn_clf_model.predict(new_features_scaled)[0]
        try:
            prediction_proba = knn_clf_model.predict_proba(new_features_scaled)[0]
        except:
            prediction_proba = None
        
        print(f"\n=== New Prediction (KNN Classification) ===")
        print(f"Input: x = {x_value[0]:.6f}, Q² = {Q2_value[0]:.6f}")
        print(f"Predicted class: {prediction_class}")
        if prediction_proba is not None:
            print(f"Class probabilities: {prediction_proba}")
        
        # Convert to continuous value using bin midpoints
        bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        continuous_value = bin_midpoints[prediction_class] if prediction_class < len(bin_midpoints) else bin_midpoints[-1]
        print(f"Approximate F2 value: {continuous_value:.6f}")
        
        return continuous_value

# Example predictions
print("\n=== Example Predictions (KNN) ===")

# First sample (regression)
sample_1_x = df["x"].iloc[0:1].values
sample_1_Q2 = df["Q^2"].iloc[0:1].values
pred_1 = predict_new_point(sample_1_x, sample_1_Q2, regression=True)
print(f"Actual F2: {df['F2_exp'].iloc[0]:.6f}")

# 500th sample (classification)
sample_500_x = df["x"].iloc[500:501].values
sample_500_Q2 = df["Q^2"].iloc[500:501].values
pred_500 = predict_new_point(sample_500_x, sample_500_Q2, regression=False)
print(f"Actual F2: {df['F2_exp'].iloc[500]:.6f}")

# Custom point (regression)
custom_x = np.array([0.1])
custom_Q2 = np.array([10.0])
predict_new_point(custom_x, custom_Q2, regression=True)

# Save results to CSV
results_df = pd.DataFrame({
    'x': df['x'],
    'Q^2': df['Q^2'],
    'logQ2': df['logQ2'],
    'F2_actual': df['F2_exp'],
    'F2_predicted_regression': y_pred_all_reg,
    'F2_predicted_class': knn_clf_model.predict(X_scaled_all),
    'error': df['F2_exp'] - y_pred_all_reg,
    'abs_error': np.abs(df['F2_exp'] - y_pred_all_reg),
    'relative_error': (df['F2_exp'] - y_pred_all_reg) / (df['F2_exp'] + 1e-10),
    'F2_category': pd.qcut(df['F2_exp'], q=num_bins, labels=False),
    'F2_binary': (df['F2_exp'] > np.median(df['F2_exp'])).astype(int)
})

results_df.to_csv('knn_predictions.csv', index=False)
print("\nPredictions saved to 'knn_predictions.csv'")

# Final comparison
print("\n" + "="*60)
print("KNN MODEL PERFORMANCE SUMMARY")
print("="*60)

print(f"\nREGRESSION METRICS:")
print(f"  Test R² Score: {test_r2:.6f}")
print(f"  Test MAE: {test_mae:.6f}")
print(f"  Test RMSE: {test_rmse:.6f}")
print(f"  Test MAPE: {test_mape:.2f}%")

print(f"\nCLASSIFICATION METRICS (Multi-class):")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision (weighted): {precision:.4f}")
print(f"  Recall (weighted): {recall:.4f}")
print(f"  F1-Score (weighted): {f1:.4f}")

print(f"\nCLASSIFICATION METRICS (Binary):")
print(f"  Accuracy: {accuracy_binary:.4f}")
print(f"  Precision: {precision_binary:.4f}")
print(f"  Recall: {recall_binary:.4f}")
print(f"  F1-Score: {f1_binary:.4f}")

print(f"\nMODEL DETAILS:")
print(f"  Number of neighbors: {knn_reg_model.n_neighbors}")
print(f"  Weighting method: {knn_reg_model.weights}")
print(f"  Distance metric: {knn_reg_model.metric}")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print("="*60)
# ======================================================
# FINAL SUMMARY TABLES (KNN) – SAFE & THESIS READY
# ======================================================

import os
from matplotlib.backends.backend_pdf import PdfPages

print("\n>>> STARTING FINAL TABLE EXPORT <<<")

# =========================
# 1. CLASSIFICATION TABLE
# =========================

classification_table = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy_binary,
        precision_binary,
        recall_binary,
        f1_binary,
        roc_auc_binary if roc_available_binary else np.nan
    ]
})

print("\n=== Classification Metrics Table ===")
print(classification_table)

# Save to Excel
classification_table.to_excel(
    "classification_metrics_knn.xlsx",
    index=False
)

# Save to PDF
with PdfPages("classification_metrics_knn.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.table(
        cellText=classification_table.values,
        colLabels=classification_table.columns,
        cellLoc='center',
        loc='center'
    )
    ax.set_title("Classification Metrics (KNN – Binary)", fontsize=14)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# =========================
# 2. REGRESSION TABLE
# =========================

regression_table = pd.DataFrame({
    "Metric": [
        "Training MAE", "Test MAE",
        "Training MSE", "Test MSE",
        "Training RMSE", "Test RMSE",
        "Training R2", "Test R2"
    ],
    "Value": [
        train_mae, test_mae,
        train_mse, test_mse,
        train_rmse, test_rmse,
        train_r2, test_r2
    ]
})

print("\n=== Regression Metrics Table ===")
print(regression_table)

# Save to Excel
regression_table.to_excel(
    "regression_metrics_knn.xlsx",
    index=False
)

# Save to PDF
with PdfPages("regression_metrics_knn.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    ax.table(
        cellText=regression_table.values,
        colLabels=regression_table.columns,
        cellLoc='center',
        loc='center'
    )
    ax.set_title("Regression Metrics (KNN)", fontsize=14)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print("\n✅ FINAL TABLES SAVED SUCCESSFULLY")
print("📂 Saved in directory:", os.getcwd())


