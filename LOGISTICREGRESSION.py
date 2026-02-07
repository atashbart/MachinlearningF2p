import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Scale features (Logistic Regression requires feature scaling)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# For comparison, also scale the entire dataset for final plots
X_scaled_all = scaler_X.transform(X)

# ================================
# APPROACH 1: MULTI-CLASS CLASSIFICATION
# ================================
print("\n" + "="*60)
print("APPROACH 1: MULTI-CLASS CLASSIFICATION")
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

print("\n=== Training Logistic Regression Model (Multi-class) ===")

# Method 1: Basic Logistic Regression with default parameters
print("Training basic Logistic Regression model...")
lr_basic = LogisticRegression(random_state=42, max_iter=1000)
lr_basic.fit(X_train_scaled, y_train_class)

# Method 2: Tuned Logistic Regression
print("\nTraining tuned Logistic Regression model...")
# Use solver that supports multi-class
lr_model = LogisticRegression(
    C=1.0,                    # Inverse of regularization strength
    penalty='l2',             # Regularization type
    solver='lbfgs',           # Optimization algorithm (supports multi-class)
    max_iter=1000,            # Maximum iterations
    random_state=42
)
lr_model.fit(X_train_scaled, y_train_class)

# Predictions
y_pred_train_class = lr_model.predict(X_train_scaled)
y_pred_test_class = lr_model.predict(X_test_scaled)
y_pred_proba_train = lr_model.predict_proba(X_train_scaled)
y_pred_proba_test = lr_model.predict_proba(X_test_scaled)

# ================================
# 1. CLASSIFICATION METRICS
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
except:
    roc_auc = "Not available (requires binary or specific encoding)"

print("\n=== Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-Score (weighted): {f1:.4f}")
if isinstance(roc_auc, float):
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
plt.title('Confusion Matrix (Logistic Regression)')
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
# 2. BINARY CLASSIFICATION
# ================================
print("\n" + "="*60)
print("APPROACH 2: BINARY CLASSIFICATION")
print("="*60)

# Convert to binary classification based on median
median_f2 = np.median(y)
y_train_binary = (y_train > median_f2).astype(int)
y_test_binary = (y_test > median_f2).astype(int)

print(f"Threshold: Median F2 = {median_f2:.4f}")
print(f"Class distribution (binary):")
print(f"  Class 0 (Low F2): {sum(y_test_binary == 0)} samples")
print(f"  Class 1 (High F2): {sum(y_test_binary == 1)} samples")

print("\n=== Training Logistic Regression Model (Binary) ===")
lr_binary = LogisticRegression(
    C=1.0,
    penalty='l2',
    max_iter=1000,
    random_state=42
)
lr_binary.fit(X_train_scaled, y_train_binary)

# Predictions for binary
y_pred_train_binary = lr_binary.predict(X_train_scaled)
y_pred_test_binary = lr_binary.predict(X_test_scaled)
y_pred_proba_train_binary = lr_binary.predict_proba(X_train_scaled)
y_pred_proba_test_binary = lr_binary.predict_proba(X_test_scaled)

# Binary classification metrics
accuracy_binary = accuracy_score(y_test_binary, y_pred_test_binary)
precision_binary = precision_score(y_test_binary, y_pred_test_binary, zero_division=0)
recall_binary = recall_score(y_test_binary, y_pred_test_binary, zero_division=0)
f1_binary = f1_score(y_test_binary, y_pred_test_binary, zero_division=0)
try:
    roc_auc_binary = roc_auc_score(y_test_binary, y_pred_proba_test_binary[:, 1])
    roc_available = True
except:
    roc_available = False

print("\n=== Binary Model Evaluation ===")
print(f"Accuracy: {accuracy_binary:.4f}")
print(f"Precision: {precision_binary:.4f}")
print(f"Recall: {recall_binary:.4f}")
print(f"F1-Score: {f1_binary:.4f}")
if roc_available:
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
plt.title('Confusion Matrix (Logistic Regression)')
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
plt.savefig('binary.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ================================
# 3. REGRESSION METRICS (via class probabilities)
# ================================
print("\n" + "="*60)
print("REGRESSION APPROXIMATION (via class probabilities)")
print("="*60)

# Convert class predictions back to continuous values using bin midpoints
def class_to_continuous(y_pred_class, bin_edges):
    """Convert class labels back to continuous values using bin midpoints"""
    continuous_vals = []
    bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
    
    for class_label in y_pred_class:
        if class_label < len(bin_midpoints):
            continuous_vals.append(bin_midpoints[class_label])
        else:
            continuous_vals.append(bin_midpoints[-1])  # Use last bin midpoint
    
    return np.array(continuous_vals)

# Convert predictions to continuous
y_pred_train_cont = class_to_continuous(y_pred_train_class, bin_edges)
y_pred_test_cont = class_to_continuous(y_pred_test_class, bin_edges)

# Calculate regression metrics
train_mae = mean_absolute_error(y_train, y_pred_train_cont)
test_mae = mean_absolute_error(y_test, y_pred_test_cont)
train_mse = mean_squared_error(y_train, y_pred_train_cont)
test_mse = mean_squared_error(y_test, y_pred_test_cont)
train_r2 = r2_score(y_train, y_pred_train_cont)
test_r2 = r2_score(y_test, y_pred_test_cont)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print("\n=== Regression Metrics (from classification) ===")
print(f"Training MAE: {train_mae:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Training MSE: {train_mse:.6f}")
print(f"Test MSE: {test_mse:.6f}")
print(f"Training RMSE: {train_rmse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Training R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")

# Logistic Regression Model Summary
print("\n=== Logistic Regression Model Summary ===")
print(f"C (Regularization strength): {lr_model.C}")
print(f"Penalty: {lr_model.penalty}")
try:
    print(f"Solver: {lr_model.solver}")
except:
    pass
try:
    print(f"Number of classes: {lr_model.n_classes_}")
except:
    pass
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Feature Importance (coefficients)
print("\n=== Feature Coefficients ===")
try:
    coef_shape = lr_model.coef_.shape
    if len(coef_shape) > 1:  # Multi-class
        for i, feature in enumerate(X.columns):
            print(f"{feature}:")
            for class_idx in range(coef_shape[0]):
                print(f"  Class {class_idx}: {lr_model.coef_[class_idx][i]:.6f}")
    else:  # Binary
        for i, feature in enumerate(X.columns):
            print(f"{feature}: {lr_model.coef_[0][i]:.6f}")
except:
    print("Coefficients not available in this format")

# Plot 1: Actual vs Predicted (continuous approximation)
plt.figure(figsize=(10, 8))

# Test set predictions
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_test_cont, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual F2')
plt.ylabel('Predicted F2 (from classification)')
plt.title('Test Set: Actual vs Predicted F2(Logistic Regression)')
plt.grid(True, alpha=0.3)

# Training set predictions
plt.subplot(2, 2, 2)
plt.scatter(y_train, y_pred_train_cont, alpha=0.6, color='green')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Actual F2')
plt.ylabel('Predicted F2 (from classification)')
plt.title('Training Set: Actual vs Predicted F2(Logistic Regression)')
plt.grid(True, alpha=0.3)

# Residuals plot
plt.subplot(2, 2, 3)
residuals_test = y_test - y_pred_test_cont
plt.scatter(y_pred_test_cont, residuals_test, alpha=0.6, color='red')
plt.axhline(y=0, color='k', linestyle='--', linewidth=2)
plt.xlabel('Predicted F2')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Test Set: Residuals Plot(Logistic Regression)')
plt.grid(True, alpha=0.3)

# Distribution of residuals
plt.subplot(2, 2, 4)
plt.hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals (Test Set)(Logistic Regression)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Actual vs Predicted (continuous approximation).pdf', format='pdf', bbox_inches='tight')
plt.show()


# Plot 2: ROC Curve for binary classification (if available)
if roc_available:
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba_test_binary[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Logistic Regression (AUC = {roc_auc_binary:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Binary Classification)(Logistic Regression)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
plt.savefig('ROC Curve for binary classification (if available).pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot 3: Probability distributions (for binary)
if roc_available:
    plt.figure(figsize=(12, 5))

    # Plot 3a: Probability distribution for class 0 (Low F2)
    plt.subplot(1, 2, 1)
    plt.hist(y_pred_proba_test_binary[y_test_binary == 0, 1], bins=30, alpha=0.7, 
             label='Actual Low F2', color='blue', edgecolor='black')
    plt.hist(y_pred_proba_test_binary[y_test_binary == 1, 1], bins=30, alpha=0.7, 
             label='Actual High F2', color='red', edgecolor='black')
    plt.xlabel('Predicted Probability of High F2')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution by True Class(Logistic Regression)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3b: Decision boundary visualization (for first feature)
    plt.subplot(1, 2, 2)
    # Create a mesh grid for visualization
    x_min, x_max = X_test_scaled[:, 0].min() - 0.5, X_test_scaled[:, 0].max() + 0.5
    y_min, y_max = X_test_scaled[:, 1].min() - 0.5, X_test_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict for mesh grid
    Z = lr_binary.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], 
                c=y_test_binary, edgecolors='k', cmap=plt.cm.RdBu)
    plt.xlabel('x (scaled)')
    plt.ylabel('logQ2 (scaled)')
    plt.title('Decision Boundary Visualization(Logistic Regression)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot3.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Plot 4: All data - Actual vs Predicted F2 vs x
plt.figure(figsize=(12, 8))

# Predict for all data
y_pred_all_class = lr_model.predict(X_scaled_all)
y_pred_all_cont = class_to_continuous(y_pred_all_class, bin_edges)

x_all_values = df["x"].values

plt.scatter(x_all_values, y, c='blue', alpha=0.6, s=30, label='Actual F2')
plt.scatter(x_all_values, y_pred_all_cont, c='red', alpha=0.6, s=30, label='Predicted F2 (Logistic Regression)')

plt.xscale('log')
plt.xlim(0.01, 1)
plt.xlabel('x (log scale)')
plt.ylabel('F2')
plt.title('All Data: Actual vs Predicted F2 vs x (Logistic Regression)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('logistic_regression_scatter_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Plot 5: Error analysis by x and Q²
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error vs x (test set)
test_indices = y_test.index
x_test = df.loc[test_indices, 'x'].values
Q2_test = df.loc[test_indices, 'Q^2'].values
errors = np.abs(y_test.values - y_pred_test_cont)

# Absolute error vs x
axes[0, 0].scatter(x_test, errors, alpha=0.6)
axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel('x (log scale)')
axes[0, 0].set_ylabel('Absolute Error')
axes[0, 0].set_title('Absolute Error vs x (Test Set)(Logistic Regression)')
axes[0, 0].grid(True, alpha=0.3)

# Absolute error vs Q²
axes[0, 1].scatter(Q2_test, errors, alpha=0.6)
axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('Q² (log scale)')
axes[0, 1].set_ylabel('Absolute Error')
axes[0, 1].set_title('Absolute Error vs Q² (Test Set)(Logistic Regression)')
axes[0, 1].grid(True, alpha=0.3)

# Relative error vs x
relative_errors = errors / (y_test.values + 1e-10)
axes[1, 0].scatter(x_test, relative_errors * 100, alpha=0.6)
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('x (log scale)')
axes[1, 0].set_ylabel('Relative Error (%)')
axes[1, 0].set_title('Relative Error vs x (Test Set)(Logistic Regression)')
axes[1, 0].grid(True, alpha=0.3)

# Distribution of errors
axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Absolute Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Absolute Errors (Test Set)(Logistic Regression)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot5.pdf', format='pdf', bbox_inches='tight')
plt.show()


# Cross-validation for Logistic Regression
print("\n=== Performing Cross-Validation (Multi-class) ===")
cv_scores = cross_val_score(lr_model, X_scaled_all, pd.qcut(y, q=num_bins, labels=False), 
                           cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.6f}")
print(f"CV Accuracy std: {cv_scores.std():.6f}")

# Cross-validation for binary
print("\n=== Performing Cross-Validation (Binary) ===")
cv_scores_binary = cross_val_score(lr_binary, X_scaled_all, (y > median_f2).astype(int), 
                                  cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy scores: {cv_scores_binary}")
print(f"Mean CV Accuracy: {cv_scores_binary.mean():.6f}")
print(f"CV Accuracy std: {cv_scores_binary.std():.6f}")

# Make predictions for specific points
def predict_new_point(x_value, Q2_value, binary=False):
    # Create feature array
    logQ2 = np.log(Q2_value)
    new_features = np.array([[x_value[0], logQ2[0]]])
    
    # Scale features using the same scaler
    new_features_scaled = scaler_X.transform(new_features)
    
    if binary:
        # Binary prediction
        prediction_class = lr_binary.predict(new_features_scaled)[0]
        prediction_proba = lr_binary.predict_proba(new_features_scaled)[0]
        
        print(f"\n=== New Prediction (Logistic Regression - Binary) ===")
        print(f"Input: x = {x_value[0]:.6f}, Q² = {Q2_value[0]:.6f}")
        print(f"Predicted class: {'High F2' if prediction_class == 1 else 'Low F2'}")
        if roc_available:
            print(f"Probability of High F2: {prediction_proba[1]:.4f}")
            print(f"Probability of Low F2: {prediction_proba[0]:.4f}")
        
        # Convert to continuous value
        if prediction_class == 0:
            continuous_value = np.median(y[y <= median_f2])
        else:
            continuous_value = np.median(y[y > median_f2])
        print(f"Approximate F2 value: {continuous_value:.6f}")
        
        return continuous_value
    else:
        # Multi-class prediction
        prediction_class = lr_model.predict(new_features_scaled)[0]
        try:
            prediction_proba = lr_model.predict_proba(new_features_scaled)[0]
        except:
            prediction_proba = None
        
        print(f"\n=== New Prediction (Logistic Regression - Multi-class) ===")
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
print("\n=== Example Predictions (Logistic Regression) ===")

# First sample (binary)
sample_1_x = df["x"].iloc[0:1].values
sample_1_Q2 = df["Q^2"].iloc[0:1].values
pred_1 = predict_new_point(sample_1_x, sample_1_Q2, binary=True)
print(f"Actual F2: {df['F2_exp'].iloc[0]:.6f}")

# 500th sample (multi-class)
sample_500_x = df["x"].iloc[500:501].values
sample_500_Q2 = df["Q^2"].iloc[500:501].values
pred_500 = predict_new_point(sample_500_x, sample_500_Q2, binary=False)
print(f"Actual F2: {df['F2_exp'].iloc[500]:.6f}")

# Custom point (binary)
custom_x = np.array([0.1])
custom_Q2 = np.array([10.0])
predict_new_point(custom_x, custom_Q2, binary=True)

# Save results to CSV
results_df = pd.DataFrame({
    'x': df['x'],
    'Q^2': df['Q^2'],
    'logQ2': df['logQ2'],
    'F2_actual': df['F2_exp'],
    'F2_predicted_class': y_pred_all_class,
    'F2_predicted_continuous': y_pred_all_cont,
    'error': df['F2_exp'] - y_pred_all_cont,
    'abs_error': np.abs(df['F2_exp'] - y_pred_all_cont),
    'relative_error': (df['F2_exp'] - y_pred_all_cont) / (df['F2_exp'] + 1e-10),
    'F2_category': pd.qcut(df['F2_exp'], q=num_bins, labels=False),
    'F2_binary': (df['F2_exp'] > np.median(df['F2_exp'])).astype(int)
})

results_df.to_csv('logistic_regression_predictions.csv', index=False)
print("\nPredictions saved to 'logistic_regression_predictions.csv'")

# Final comparison
print("\n" + "="*60)
print("LOGISTIC REGRESSION MODEL PERFORMANCE SUMMARY")
print("="*60)

print(f"\nMULTI-CLASS CLASSIFICATION METRICS:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision (weighted): {precision:.4f}")
print(f"  Recall (weighted): {recall:.4f}")
print(f"  F1-Score (weighted): {f1:.4f}")

print(f"\nBINARY CLASSIFICATION METRICS:")
print(f"  Accuracy: {accuracy_binary:.4f}")
print(f"  Precision: {precision_binary:.4f}")
print(f"  Recall: {recall_binary:.4f}")
print(f"  F1-Score: {f1_binary:.4f}")
if roc_available:
    print(f"  ROC-AUC: {roc_auc_binary:.4f}")

print(f"\nREGRESSION METRICS (from classification):")
print(f"  Test R²: {test_r2:.6f}")
print(f"  Test MAE: {test_mae:.6f}")
print(f"  Test RMSE: {test_rmse:.6f}")

print(f"\nMODEL DETAILS:")
try:
    print(f"  Number of classes: {lr_model.n_classes_}")
except:
    pass
print(f"  Regularization (C): {lr_model.C}")
print(f"  Penalty type: {lr_model.penalty}")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print("="*60)

# Additional: Feature importance analysis
print("\n=== Feature Importance Analysis ===")
print("Positive coefficients increase probability of higher classes")
print("Negative coefficients decrease probability of higher classes")

try:
    if len(lr_model.coef_.shape) > 1:
        for i, feature in enumerate(X.columns):
            avg_coef = np.mean(lr_model.coef_[:, i])
            print(f"{feature}: Average coefficient = {avg_coef:.6f}")
    else:
        for i, feature in enumerate(X.columns):
            print(f"{feature}: Coefficient = {lr_model.coef_[0][i]:.6f}")
except:
    print("Feature coefficients not available")
  # Use the best model
    dt_best = grid_search.best_estimator_
    y_pred_test_best = dt_best.predict(X_test_scaled)
    test_r2_best = r2_score(y_test, y_pred_test_best)
    print(f"Test R² with best model: {test_r2_best:.6f}")
    print(f"Improvement: {test_r2_best - test_r2:.6f}")
# Make predictions (class labels and probabilities)

# ============================================
# FINAL TABLE 1: CLASSIFICATION METRICS TABLE
# ============================================

classification_metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],
    "Multi-class (weighted)": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc if isinstance(roc_auc, float) else np.nan
    ],
    "Binary": [
        accuracy_binary,
        precision_binary,
        recall_binary,
        f1_binary,
        roc_auc_binary if roc_available else np.nan
    ]
})

print("\n=== Classification Metrics Table ===")
print(classification_metrics_df)

# ---- Save to Excel ----
classification_metrics_df.to_excel(
    "classification_metrics_logistic_regression.xlsx",
    index=False
)

# ---- Save to PDF ----
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=classification_metrics_df.round(4).values,
    colLabels=classification_metrics_df.columns,
    loc='center'
)
table.scale(1, 1.5)
plt.title("Classification Metrics (Logistic Regression)", fontsize=12)
plt.savefig(
    "classification_metrics_logistic_regression.pdf",
    bbox_inches='tight'
)
plt.close()
# ============================================
# FINAL TABLE 2: REGRESSION METRICS TABLE
# ============================================

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

print("\n=== Regression Metrics Table ===")
print(regression_metrics_df)

# ---- Save to Excel ----
regression_metrics_df.to_excel(
    "regression_metrics_logistic_regression.xlsx",
    index=False
)

# ---- Save to PDF ----
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
table = ax.table(
    cellText=regression_metrics_df.round(6).values,
    colLabels=regression_metrics_df.columns,
    loc='center'
)
table.scale(1, 1.5)
plt.title("Regression Metrics (Logistic Regression)", fontsize=12)
plt.savefig(
    "regression_metrics_logistic_regression.pdf",
    bbox_inches='tight'
)
plt.close()


