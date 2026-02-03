# ==========================================
# Random Forest F1-score Analysis (FINAL)
# Regression → 3-Class Classification
# Dataset: F2BCMS.csv
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

print("=== Random Forest F1-score Analysis ===")

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
y = df["F2_exp"]

# -------------------------------------------------
# 3. Train / Test split (NO leakage)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# -------------------------------------------------
# 4. Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# 5. Regression → Classification (Quantiles)
# -------------------------------------------------
def create_classes(y, n_classes=3):
    percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
    thresholds = np.percentile(y, percentiles)
    return np.digitize(y, thresholds)

y_train_class = create_classes(y_train, 3)
y_test_class = create_classes(y_test, 3)

print("\nClass distribution (training):")
classes, counts = np.unique(y_train_class, return_counts=True)
for c, n in zip(classes, counts):
    print(f"  Class {c}: {n} samples ({n/len(y_train_class)*100:.1f}%)")

# -------------------------------------------------
# 6. Random Forest + F1-score
# -------------------------------------------------
n_estimators_values = range(10, 301, 10)

train_f1_scores = []
cv_f1_scores = []
cv_f1_std = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining Random Forest models...")

for i, n_est in enumerate(n_estimators_values, start=1):

    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1          # ✅ parallel فقط اینجا
    )

    # Training F1
    rf.fit(X_train_scaled, y_train_class)
    y_train_pred = rf.predict(X_train_scaled)
    train_f1_scores.append(
        f1_score(y_train_class, y_train_pred, average="weighted")
    )

    # Cross-validation F1 (NO parallel here → stable)
    cv_scores = cross_val_score(
        rf,
        X_train_scaled,
        y_train_class,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=1           # ✅ مهم: جلوگیری از crash
    )

    cv_f1_scores.append(cv_scores.mean())
    cv_f1_std.append(cv_scores.std())

    if i % 5 == 0:
        print(f"  {i}/{len(n_estimators_values)} models trained")

# -------------------------------------------------
# 7. Optimal model
# -------------------------------------------------
optimal_idx = np.argmax(cv_f1_scores)
optimal_n_estimators = list(n_estimators_values)[optimal_idx]

# -------------------------------------------------
# 8. Plot (Journal / Thesis style)
# -------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(
    n_estimators_values,
    train_f1_scores,
    label="Training F1-score",
    color="blue",
    linewidth=2.5,
    marker="o",
    markevery=5
)

plt.plot(
    n_estimators_values,
    cv_f1_scores,
    label="Cross-validation F1-score",
    color="red",
    linewidth=2.5,
    marker="s",
    markevery=5
)

plt.fill_between(
    n_estimators_values,
    np.array(cv_f1_scores) - np.array(cv_f1_std),
    np.array(cv_f1_scores) + np.array(cv_f1_std),
    alpha=0.2,
    color="red",
    label="CV ± 1 std"
)

plt.axvline(
    optimal_n_estimators,
    linestyle="--",
    color="green",
    linewidth=2,
    label=f"Optimal n_estimators = {optimal_n_estimators}"
)

plt.plot(
    optimal_n_estimators,
    cv_f1_scores[optimal_idx],
    "go",
    markersize=10,
    markeredgecolor="black"
)

plt.xlabel("Number of Trees (n_estimators)", fontsize=12, fontweight="bold")
plt.ylabel("F1-score (weighted)", fontsize=12, fontweight="bold")
plt.title(
    "Random Forest Classifier\n"
    "F1-score vs Number of Trees (3-class classification)",
    fontsize=14,
    fontweight="bold"
)

plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()

# ✅ EXACT filenames you asked for
plt.savefig("72.pdf", dpi=300, bbox_inches="tight")
plt.savefig("f1_score_plot_from_regression.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------
# 9. Summary
# -------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Number of classes      : {len(np.unique(y_train_class))}")
print(f"Optimal n_estimators   : {optimal_n_estimators}")
print(f"Best CV F1-score       : {cv_f1_scores[optimal_idx]:.4f}")
print(f"Training F1 (optimal)  : {train_f1_scores[optimal_idx]:.4f}")
print("=" * 60)
