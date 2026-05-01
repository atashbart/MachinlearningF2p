# ============================================================
# GBOOST | GPR | MLP | SVC – F1-score Hyperparameter Analysis
# FIXED: GPR runs on a small subset to avoid O(n^3) hang
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# ---------- Data loading & preparation (unchanged) ----------
df = pd.read_csv("F2BCMS.csv")

df["logQ2"] = np.log(df["Q^2"])
df["logx"]  = np.log(df["x"].clip(lower=1e-6))

X = df[["logx", "logQ2"]]
y = df["F2_exp"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

def create_classes(y, n_classes=3):
    percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
    thresholds = np.percentile(y, percentiles)
    return np.digitize(y, thresholds)

y_train_class = create_classes(y_train, 3)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- Gradient Boosting (n_estimators) ----------
print("1/4 Gradient Boosting ...")
gb_n = range(10, 301, 10)
gb_train, gb_cv, gb_std = [], [], []

for n in gb_n:
    gb = GradientBoostingClassifier(
        n_estimators=n,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train_class)
    gb_train.append(
        f1_score(y_train_class, gb.predict(X_train_scaled), average="weighted")
    )
    scores = cross_val_score(
        gb, X_train_scaled, y_train_class,
        cv=cv, scoring="f1_weighted", n_jobs=1
    )
    gb_cv.append(scores.mean())
    gb_std.append(scores.std())

# ---------- Gaussian Process (RBF kernel length_scale) ----------
# *** FIX: use a random subset to avoid O(n^3) hang ***
print("2/4 Gaussian Process (on subset) ...")
subset_size = min(500, len(X_train_scaled))
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train_scaled), size=subset_size, replace=False)
X_gp = X_train_scaled[idx]
y_gp = y_train_class[idx]

ls_values = np.logspace(-2, 1, 10)      # کاهش تعداد نقاط برای سرعت
gp_train, gp_cv, gp_std = [], [], []

for ls in ls_values:
    kernel = RBF(length_scale=ls)
    gp = GaussianProcessClassifier(
        kernel=kernel,
        optimizer=None,                # keep kernel fixed
        random_state=42
    )
    gp.fit(X_gp, y_gp)
    gp_train.append(
        f1_score(y_gp, gp.predict(X_gp), average="weighted")
    )
    # اعتبارسنجی روی همان زیرمجموعه
    scores = cross_val_score(
        gp, X_gp, y_gp,
        cv=cv, scoring="f1_weighted", n_jobs=1
    )
    gp_cv.append(scores.mean())
    gp_std.append(scores.std())

# ---------- MLP (regularization alpha) ----------
print("3/4 MLP ...")
alpha_values = np.logspace(-3, 3, 20)
mlp_train, mlp_cv, mlp_std = [], [], []

for a in alpha_values:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        alpha=a,
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train_class)
    mlp_train.append(
        f1_score(y_train_class, mlp.predict(X_train_scaled), average="weighted")
    )
    scores = cross_val_score(
        mlp, X_train_scaled, y_train_class,
        cv=cv, scoring="f1_weighted", n_jobs=1
    )
    mlp_cv.append(scores.mean())
    mlp_std.append(scores.std())

# ---------- SVR  ----------
print("4/4 SVR ...")
C_values = np.logspace(-3, 3, 20)
svc_train, svc_cv, svc_std = [], [], []

for C in C_values:
    svc = SVC(
        C=C,
        kernel="rbf",
        gamma="scale",
        random_state=42
    )
    svc.fit(X_train_scaled, y_train_class)
    svc_train.append(
        f1_score(y_train_class, svc.predict(X_train_scaled), average="weighted")
    )
    scores = cross_val_score(
        svc, X_train_scaled, y_train_class,
        cv=cv, scoring="f1_weighted", n_jobs=1
    )
    svc_cv.append(scores.mean())
    svc_std.append(scores.std())

# ---------- Plot (2x2 layout) ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# (A) Gradient Boosting
ax[0,0].plot(gb_n, gb_train, 'o-', label="Train")
ax[0,0].plot(gb_n, gb_cv, 's-', label="CV")
ax[0,0].fill_between(gb_n, np.array(gb_cv)-gb_std, np.array(gb_cv)+gb_std, alpha=0.2)
ax[0,0].set_title("(A) Gradient Boosting", fontweight="bold")
ax[0,0].set_xlabel("n_estimators")
ax[0,0].set_ylabel("F1-score")
ax[0,0].legend()
ax[0,0].grid(alpha=0.3)

# (B) Gaussian Process
ax[0,1].semilogx(ls_values, gp_train, 'o-', label="Train (subset)")
ax[0,1].semilogx(ls_values, gp_cv, 's-', label="CV (subset)")
ax[0,1].fill_between(ls_values, np.array(gp_cv)-gp_std, np.array(gp_cv)+gp_std, alpha=0.2)
ax[0,1].set_title("(B) Gaussian Process (RBF length_scale)", fontweight="bold")
ax[0,1].legend()
ax[0,1].grid(alpha=0.3)

# (C) MLP
ax[1,0].semilogx(alpha_values, mlp_train, 'o-', label="Train")
ax[1,0].semilogx(alpha_values, mlp_cv, 's-', label="CV")
ax[1,0].fill_between(alpha_values, np.array(mlp_cv)-mlp_std, np.array(mlp_cv)+mlp_std, alpha=0.2)
ax[1,0].set_title("(C) MLP ($\\alpha$)", fontweight="bold")
ax[1,0].set_xlabel("alpha")
ax[1,0].set_ylabel("F1-score")
ax[1,0].legend()
ax[1,0].grid(alpha=0.3)

# (D) SVR
ax[1,1].semilogx(C_values, svc_train, 'o-', label="Train")
ax[1,1].semilogx(C_values, svc_cv, 's-', label="CV")
ax[1,1].fill_between(C_values, np.array(svc_cv)-svc_std, np.array(svc_cv)+svc_std, alpha=0.2)
ax[1,1].set_title("(D) SVR (C)", fontweight="bold")
ax[1,1].set_xlabel("C")
ax[1,1].legend()
ax[1,1].grid(alpha=0.3)

plt.suptitle(
    "F1-score Hyperparameter Optimization\nGradient Boosting | Gaussian Process | MLP | SVC",
    fontsize=16, fontweight="bold"
)

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("F1_4panel_GB_GP_MLP_SVR.pdf", dpi=300, bbox_inches="tight")
plt.savefig("F1_4panel_GB_GP_MLP_SVR.png", dpi=300, bbox_inches="tight")
plt.show()
