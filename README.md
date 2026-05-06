# Machine Learning for Predicting the Proton Structure Function F₂ in QCD

**Shahin Atashbar Tehrani¹² and Elham Astaraki³**

¹ School of Particles and Accelerators, Institute for Research in Fundamental Sciences (IPM), Tehran, Iran  
² Department of Physics, Faculty of Nano and Bio Science and Technology, Persian Gulf University, Bushehr, Iran  
³ Department of Physics, Razi University, Kermanshah, Iran  

---

## Overview

This repository provides a machine learning framework for predicting the proton structure function \(F_2(x, Q^2)\) using experimental deep‑inelastic scattering (DIS) data.  
The project includes both direct regression and optional quantile‑based classification for extended analysis.

---

## Dataset

- File: `F2BCMS.csv`  
- Inputs: `x`, `Q^2` (log‑transformed to `logx`, `logQ2`)  
- Target: `F2_exp`

---

## Models

- Support Vector Regression (SVR)  
- Gradient Boosting Regressor (GBoost)  
- Multi‑Layer Perceptron (MLP)  
- Gaussian Process Regression (GPR)  

---

## Methodology

- Logarithmic feature transformation  
- StandardScaler preprocessing  
- 80/20 train–test split  
- Cross‑validated hyperparameter tuning  
- Optional quantile‑based classification (3 or 9 classes)

---

## Evaluation

**Regression:** MAE, MSE, RMSE, \(R^2\)  
**Classification (optional):** Accuracy, Precision, Recall, F1, ROC–AUC

---

## Outputs

- Publication‑ready plots (`.png`, `.pdf`)  
- Summary tables (`.xlsx`)  
- Learning curves, residuals  
- Optional confusion matrices

---

## Repository Structure
├── F2BCMS.csv

├── svr.py

├── gboost.py

├── NN.py

├── GPR.py

├── f1.py

├── plots/

├── tables/

└── RE
---

## Running

---

## Citation

S. Atashbar Tehrani and E. Astaraki,  
*Machine Learning for Predicting the Proton Structure Function F₂ in QCD*,  
Manuscript in preparation / under review.

---

## Contact

**Shahin Atashbar Tehrani** — atashbart3@gmail.com  
**Elham Astaraki** — astaraki.elham@razi.ac.ir

