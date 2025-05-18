# MOML Project: IMT2022050_118

## Part 1: Fairness-Aware Classification on the Bank Marketing Dataset

### 📄 Dataset Overview

The dataset used for this part of the project is the [Bank Marketing Dataset]([https://archive.ics.uci.edu/ml/datasets/bank+marketing](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)), This is the classic marketing bank dataset uploaded originally in the UCI Machine Learning Repository. The dataset gives you information about a marketing campaign of a financial institution in which you will have to analyze in order to find ways to look for future strategies in order to improve future marketing campaigns for the bank. Each row corresponds to a client and contains attributes such as:

- **age**: Age of the client
- **job**: Type of job (e.g., admin, technician, etc.)
- **marital**: Marital status (`single`, `married`, `divorced`)
- **education**: Education completed (`secondary`, `tertiary`, etc)
- **default**: Has credit in default? (`yes`/`no`)
- **balance**: Average yearly balance in euros
- **housing**: Has housing? (`yes`/`no`)
- **loan**: Has personal loan? (`yes`/`no`)
- **contact**: Contact communication type ('cellular', etc.)
- **day**: Last contact day of the month

### 🧠 Problem Statement

We aim to train a binary classifier that predicts whether a client will subscribe to a term deposit (`deposit = yes` or `no`). However, **our primary focus is to ensure fairness in predictions with respect to the `marital` attribute**, particularly between **"single"** and **"divorced"** individuals.

We exclude the "married" category to isolate the fairness analysis to single vs divorced, as any potential bias between those two is not logically grounded (unlike married vs others which could involve financial co-dependence).

---

## 🧪 Methodology

### 1. **Data Preprocessing**
- Dropped missing values.
- Filtered the dataset to include only `marital` values of `single` or `divorced`.
- Converted categorical variables using one-hot encoding.
- Binary columns like `default`, `housing`, and `loan` were mapped to 0/1.
- Target column `deposit` was mapped to 0 (no) and 1 (yes).
- `marital` was encoded as: `divorced` = 1, `single` = 0 (for fairness computation).
- Train-test split performed with stratification and standard scaling.

### 2. **Fairness Metric: Demographic Parity Difference (DPD)**
We compute DPD as:

DPD = |P(ŷ = 1 | marital = divorced) - P(ŷ = 1 | marital = single)|


This measures the absolute difference in positive prediction rates between divorced and single individuals.

### 3. **Model Selection via Bayesian Optimization**
We perform hyperparameter optimization of a `RandomForestClassifier` using `BayesSearchCV` from `skopt`. The search space includes:
- `n_estimators`: [50, 300]
- `max_depth`: [5, 30]
- `min_samples_split`: [2, 10]

The best model is selected based on 3-fold cross-validated accuracy.

### 4. **Threshold Tuning with Fairness-Accuracy Trade-off**
- Generate prediction probabilities on the test set.
- Evaluate across a grid of decision thresholds and alpha values.
- For each pair `(threshold, alpha)`, compute:
  - Accuracy
  - DPD
  - Objective = `alpha * DPD + (1 - alpha) * (1 - accuracy)`
- This allows control over the trade-off between fairness and accuracy.

### 5. **Pareto Frontier Filtering**
- Identify Pareto-optimal points where neither accuracy nor DPD can be improved without worsening the other.
- These points represent the best trade-offs between fairness and predictive performance.

### 6. **Visualization**
- Plotted the **Pareto frontier** showing the trade-off between DPD and accuracy.
- A strictly non-dominated set of points was visualized using `matplotlib`.

---

## 📈 Key Output

- **Strict Pareto Frontier Points**: Highlights the number of optimal fairness-accuracy trade-off points.
- **EDA Insight**:
  - Percentage of `deposit = yes` for `divorced` vs `single` shows existing bias in the raw dataset.

---

## 🧮 Conclusion

This part of the project evaluates and mitigates potential **unjustified bias** in classification outcomes between clients who are **divorced** and **single**. It applies a **Bayesian-optimized Random Forest model** and a fairness-aware thresholding strategy, demonstrating how predictive performance can be aligned with fairness objectives.

---

## 📂 Next Section Placeholder

> **Coming Up:**
> **Part 2: [Your Section Title Here]**
>
> _[Short summary of what the next part will contain, such as another dataset, deep learning models, causal analysis, etc.]_

---

### 👨‍💻 Author

- **Name**: IMT2022050_118
- **Course**: MOML - Machine Learning for Observational Data
- **Institution**: [Insert Institution Name Here]
