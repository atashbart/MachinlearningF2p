# ==========================================
# KNN F1-score Analysis (FINAL – THESIS READY)
# Dataset: F2BCMS.csv
# Regression → 3-Class Classification
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

print("=== KNN F1-score Analysis ===")

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
df = pd.read_csv("F2BCMS.csv")
print(df.head())
print(f"Dataset shape: {df.shape}")

# -------------------------------------------------
# 2. Feature engineering
# -------------------------------------------------
df["logQ2"] = np.log(df["Q^2"])
df["logx"] = np.log(df["x"].clip(lower=1e-6))

X = df[["logx", "logQ2"]]
y = df["F2_exp"]  # continuous target

# -------------------------------------------------
# 3. Train / Test split (NO leakage)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# -------------------------------------------------
# 4. Scaling (MANDATORY for KNN)
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# 5. Regression → 3-Class Classification
# -------------------------------------------------
def create_classes(y, n_classes=3):
    percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
    thresholds = np.percentile(y, percentiles)
    return np.digitize(y, thresholds)

y_train_class = create_classes(y_train, 3)
y_test_class = create_classes(y_test, 3)

print("\nClass distribution (training set):")
classes, counts = np.unique(y_train_class, return_counts=True)
for c, n in zip(classes, counts):
    print(f"  Class {c}: {n} samples ({n/len(y_train_class)*100:.1f}%)")

# -------------------------------------------------
# 6. KNN + F1-score analysis
# -------------------------------------------------
k_values = range(1, 51)

train_f1_scores = []
cv_f1_scores = []
cv_f1_std = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining KNN models...")

for i, k in enumerate(k_values, start=1):

    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance"   # better for physics data
    )

    # Training F1
    knn.fit(X_train_scaled, y_train_class)
    y_train_pred = knn.predict(X_train_scaled)
    train_f1_scores.append(
        f1_score(y_train_class, y_train_pred, average="weighted")
    )

    # Cross-validation F1 (NO parallel → stable)
    cv_scores = cross_val_score(
        knn,
        X_train_scaled,
        y_train_class,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=1
    )

    cv_f1_scores.append(cv_scores.mean())
    cv_f1_std.append(cv_scores.std())

    if i % 5 == 0:
        print(f"  k = {k} completed")

# -------------------------------------------------
# 7. Optimal k
# -------------------------------------------------
optimal_idx = np.argmax(cv_f1_scores)
optimal_k = list(k_values)[optimal_idx]

# -------------------------------------------------
# 8. Plot (Training vs CV F1)
# -------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(
    k_values, train_f1_scores,
    label="Training F1-score",
    color="blue",
    linewidth=2.5,
    marker="o",
    markevery=2
)

plt.plot(
    k_values, cv_f1_scores,
    label="Cross-validation F1-score",
    color="red",
    linewidth=2.5,
    marker="s",
    markevery=2
)

plt.fill_between(
    k_values,
    np.array(cv_f1_scores) - np.array(cv_f1_std),
    np.array(cv_f1_scores) + np.array(cv_f1_std),
    alpha=0.2,
    color="red",
    label="CV ± 1 std"
)

plt.axvline(
    optimal_k,
    linestyle="--",
    color="green",
    linewidth=2,
    label=f"Optimal k = {optimal_k}"
)

plt.plot(
    optimal_k,
    cv_f1_scores[optimal_idx],
    "go",
    markersize=10,
    markeredgecolor="black"
)

plt.xlabel("Number of Neighbors (k)", fontsize=12, fontweight="bold")
plt.ylabel("F1-score (weighted)", fontsize=12, fontweight="bold")
plt.title(
    "KNN Classifier\nF1-score vs Number of Neighbors (3-class classification)",
    fontsize=14,
    fontweight="bold"
)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("knn_f1_score_analysis.pdf", dpi=300, bbox_inches="tight")
plt.savefig("knn_f1_score_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------
# 9. Generalization Gap Plot
# -------------------------------------------------
plt.figure(figsize=(10, 5))

gap = np.array(train_f1_scores) - np.array(cv_f1_scores)

plt.plot(
    k_values, gap,
    color="purple",
    linewidth=2.5,
    marker="^",
    markevery=2,
    label="Generalization Gap (Train − CV)"
)

plt.axhline(0, color="black", linewidth=1)
plt.axvline(optimal_k, linestyle="--", color="green", label=f"Optimal k={optimal_k}")

plt.xlabel("Number of Neighbors (k)", fontsize=12)
plt.ylabel("Generalization Gap", fontsize=12)
plt.title("KNN Overfitting Analysis", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

plt.savefig("knn_generalization_gap.pdf", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------
# 10. Summary
# -------------------------------------------------
print("\n" + "=" * 60)
print("KNN SUMMARY")
print("=" * 60)
print(f"Number of classes        : {len(np.unique(y_train_class))}")
print(f"Optimal k                : {optimal_k}")
print(f"Best CV F1-score         : {cv_f1_scores[optimal_idx]:.4f}")
print(f"Training F1 at optimal   : {train_f1_scores[optimal_idx]:.4f}")
print(f"Generalization gap       : {train_f1_scores[optimal_idx] - cv_f1_scores[optimal_idx]:.4f}")
print("=" * 60)
