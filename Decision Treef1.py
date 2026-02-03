# ==================================================
# Decision Tree F1-score Analysis (FINAL)
# Dataset: F2BCMS.csv
# Regression → 3-Class Classification
# ==================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

print("=== Decision Tree F1-score Analysis ===")

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
# 3. Train / Test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 4. Scaling (NOT required for DT, but kept for consistency)
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
# 6. Decision Tree depth sweep
# -------------------------------------------------
max_depth_values = range(1, 31)

train_f1_scores = []
cv_f1_scores = []
cv_f1_std = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining Decision Tree models...")

for depth in max_depth_values:

    dt = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    # Training F1
    dt.fit(X_train_scaled, y_train_class)
    y_train_pred = dt.predict(X_train_scaled)
    train_f1_scores.append(
        f1_score(y_train_class, y_train_pred, average="weighted")
    )

    # Cross-validation F1
    cv_scores = cross_val_score(
        dt,
        X_train_scaled,
        y_train_class,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=1
    )

    cv_f1_scores.append(cv_scores.mean())
    cv_f1_std.append(cv_scores.std())

# -------------------------------------------------
# 7. Optimal depth
# -------------------------------------------------
optimal_idx = np.argmax(cv_f1_scores)
optimal_depth = list(max_depth_values)[optimal_idx]

# -------------------------------------------------
# 8. Plot: Training vs CV F1
# -------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(
    max_depth_values,
    train_f1_scores,
    label="Training F1-score",
    linewidth=2.5,
    marker="o"
)

plt.plot(
    max_depth_values,
    cv_f1_scores,
    label="Cross-validation F1-score",
    linewidth=2.5,
    marker="s"
)

plt.fill_between(
    max_depth_values,
    np.array(cv_f1_scores) - np.array(cv_f1_std),
    np.array(cv_f1_scores) + np.array(cv_f1_std),
    alpha=0.2,
    label="CV ± 1 std"
)

plt.axvline(
    optimal_depth,
    linestyle="--",
    color="green",
    linewidth=2,
    label=f"Optimal depth = {optimal_depth}"
)

plt.plot(
    optimal_depth,
    cv_f1_scores[optimal_idx],
    "go",
    markersize=10,
    markeredgecolor="black"
)

plt.xlabel("Maximum Tree Depth", fontsize=12, fontweight="bold")
plt.ylabel("F1-score (weighted)", fontsize=12, fontweight="bold")
plt.title("Decision Tree\nTraining vs Cross-Validation F1-score",
    fontsize=14,
    fontweight="bold"
)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("dt_f1_score_analysis.pdf", dpi=300, bbox_inches="tight")
plt.savefig("dt_f1_score_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------
# 9. Summary
# -------------------------------------------------
print("\n" + "=" * 60)
print("DECISION TREE SUMMARY")
print("=" * 60)
print(f"Number of classes        : {len(np.unique(y_train_class))}")
print(f"Optimal max_depth        : {optimal_depth}")
print(f"Best CV1-score          : {cv_f1_scores[optimal_idx]:.4f}")
print(f"Training F1 at optimal   : {train_f1_scores[optimal_idx]:.4f}")
print(f"Generalization gap       : {train_f1_scores[optimal_idx] - cv_f1_scores[optimal_idx]:.4f}")
print("=" * 60)
