# Credit Risk Assessment & MLOps Monitoring Framework

## Project Overview

This project implements a complete machine learning pipeline for credit risk classification, from exploratory data analysis to production-grade MLOps monitoring. The system is designed to predict the probability of default for customers using demographic data and payment history, with particular emphasis on model interpretability and performance monitoring in regulated fintech environments.

**Key Technologies:** Python, scikit-learn, XGBoost, LightGBM, SHAP, Statistical Testing

---

## Dataset Description

The project uses two relational datasets:

1. **Customer Data** (demographic & categorical features)
   - Demographic attributes
   - Encoded categorical features (fea_1, fea_3, fea_5, fea_6, fea_7, fea_9)
   - Binary target: `label` (0 = Low Risk, 1 = High Risk)

2. **Payment Data** (transactional history)
   - Overdue payment indicators (OVD_t1, OVD_t2, OVD_t3)
   - Payment behavior (pay_normal)
   - Balance metrics (new_balance, highest_balance)
   - Credit product information (prod_code, prod_limit)

**Key Challenge:** The dataset exhibits significant class imbalance (~88% Low Risk, ~12% High Risk), which is characteristic of real-world credit risk problems.

---

## Methodology

### Notebook 1: Exploratory Data Analysis & Preprocessing

#### 1.1 Multi-Table Data Integration

The analysis begins with separate exploration of customer and payment tables, followed by intelligent aggregation:

```
payment_agg = groupby(customer_id).agg({
    'OVD_sum': ['sum', 'mean', 'max', 'std'],
    'new_balance': ['mean', 'max', 'min', 'std'],
    ...
})
```

This aggregation strategy preserves temporal information while creating a unified customer-level view.

#### 1.2 Missing Data Treatment

- **Numerical features:** Median imputation to maintain robustness against outliers
- **Categorical features:** Mode imputation
- **Validation:** Zero missing values after imputation, preserving 100% of observations

#### 1.3 Outlier Management

Applied **Winsorization** at 1st and 99th percentiles rather than removal:

$$x_{winsorized} = \begin{cases} 
P_1 & \text{if } x < P_1 \\
P_{99} & \text{if } x > P_{99} \\
x & \text{otherwise}
\end{cases}$$

This approach maintains sample size while limiting extreme value influence on gradient-based models.

#### 1.4 Class Imbalance Analysis

The target variable exhibits a ratio of approximately 7:1 (Low Risk:High Risk), necessitating specialized handling in subsequent modeling phases.

---

### Notebook 2: Feature Engineering & Selection

#### 2.1 Domain-Driven Feature Creation

**Financial Ratios:**
- Credit Utilization: $\text{CU} = \frac{\text{current\_balance}}{\text{credit\_limit}}$
- Debt Service Coverage: $\text{DSC} = \frac{\text{normal\_payments}}{\text{total\_obligations}}$

**Risk Indicators:**
- Overdue Severity Score: $S = w_1 \cdot \text{OVD}_1 + w_2 \cdot \text{OVD}_2 + w_3 \cdot \text{OVD}_3$ 
  where $w_1=1, w_2=3, w_3=10$ (escalating penalties for severity)

**Behavioral Features:**
- Payment Reliability: $R = \frac{\text{normal\_payments}}{\text{normal\_payments} + \text{overdue\_sum} + \epsilon}$
- Balance Stability: $\sigma_{normalized} = \frac{\sigma_{balance}}{\mu_{balance} + \epsilon}$

**Interaction Terms:**
- Cross-feature products to capture non-linear relationships (e.g., overdue_rate × credit_utilization)

#### 2.2 Feature Selection Pipeline

Multi-stage selection process to balance predictive power and computational efficiency:

**Stage 1: Variance Threshold**
- Remove features with variance < 0.01
- Eliminates quasi-constant features that provide minimal information

**Stage 2: Correlation-Based Filtering**
- Pearson correlation threshold: |ρ| > 0.95
- Retention strategy: keep feature with highest target correlation
- Reduces multicollinearity while preserving predictive signal

**Stage 3: Mutual Information Ranking**

$$I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}$$

Measures non-linear dependencies between features and target variable.

**Stage 4: Random Forest Importance**
- Gini importance from ensemble of 100 trees
- Captures feature interactions and non-linear effects

**Final Selection:** Consensus ranking combining MI scores and RF importance, selecting top 30 features.

#### 2.3 Data Preprocessing

Applied **RobustScaler** for standardization:

$$z = \frac{x - \text{median}(X)}{\text{IQR}(X)}$$

More resistant to outliers than standard z-score normalization, critical given financial data characteristics.

---

### Notebook 3: Modeling & Evaluation

#### 3.1 Model Architecture Comparison

Evaluated five algorithms spanning different paradigms:

1. **Logistic Regression** (Baseline)
   - Linear decision boundary
   - L2 regularization (Ridge)
   - max_iter=5000 to ensure convergence

2. **Random Forest**
   - Ensemble of 300 decision trees
   - Gini impurity criterion
   - Bootstrap aggregation with feature subsampling

3. **Gradient Boosting**
   - Sequential tree building
   - Learning rate: 0.1
   - Maximum depth: 5 (prevents overfitting)

4. **XGBoost**
   - Regularized gradient boosting
   - L1/L2 regularization
   - scale_pos_weight parameter for class imbalance

5. **LightGBM** (Best performer)
   - Leaf-wise tree growth
   - Gradient-based One-Side Sampling (GOSS)
   - Exclusive Feature Bundling (EFB)

#### 3.2 Class Imbalance Mitigation

**Approach 1: Class Weighting**
- Inverse frequency weighting: $w_i = \frac{n_{samples}}{n_{classes} \cdot n_{samples,i}}$
- Applied in LightGBM via `class_weight='balanced'`

**Approach 2: SMOTE (Synthetic Minority Over-sampling)**
- Generates synthetic High Risk samples using k-nearest neighbors
- Sampling strategy: 0.5 (50% of majority class)
- Addresses distributional imbalance without simply duplicating samples

**Approach 3: Cost-Sensitive Learning**
- Tested class weights ranging from 1:5 to 1:30
- Optimal weight: 1:15 (balances precision/recall trade-off)

#### 3.3 Decision Threshold Optimization

The standard 0.5 classification threshold is suboptimal for imbalanced problems. Applied grid search over thresholds $\tau \in [0.05, 0.95]$ to maximize F1-score:

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

**Optimal threshold identified:** $\tau^* = 0.08$

This aggressive threshold prioritizes recall (detecting High Risk customers) at acceptable precision cost, appropriate for risk-averse lending contexts.

#### 3.4 Model Performance

**LightGBM + Optimized Threshold (Final Model):**
- ROC-AUC: 0.6475
- F1-Score: 0.4052
- Precision: 0.2870
- Recall: 0.6889

**Interpretation:**
- The model successfully identifies 69% of actual High Risk customers (recall)
- Moderate ROC-AUC reflects the inherent difficulty of credit risk prediction with limited features
- Performance aligns with typical fintech production systems (0.65-0.75 AUC range)

#### 3.5 Model Interpretability (SHAP Analysis)

Employed SHAP (SHapley Additive exPlanations) for model-agnostic interpretability:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S)]$$

Where $\phi_i$ represents the Shapley value (contribution) of feature $i$ to the prediction.

**Key Insights:**
- Payment reliability and overdue severity are dominant predictors
- Non-linear interactions captured effectively by tree-based models
- Feature contributions vary significantly across risk segments

---

### Notebook 4: MLOps Monitoring Framework

#### 4.1 Production Monitoring Architecture

Developed a `CreditRiskMonitor` class implementing comprehensive model monitoring:

**Core Components:**
1. Performance tracking (accuracy, precision, recall, F1, ROC-AUC)
2. Data drift detection via statistical testing
3. Prediction drift monitoring
4. Automated alerting system

#### 4.2 Data Drift Detection Methodology

**Population Stability Index (PSI) per feature:**

$$\text{PSI}_i = \sum_{j=1}^{n_{bins}} (\text{actual}_j - \text{expected}_j) \times \ln\left(\frac{\text{actual}_j}{\text{expected}_j}\right)$$

**Simplified implementation:**
$$\text{Drift Score} = \frac{|\mu_{current} - \mu_{reference}|}{\sigma_{reference}}$$

Features with drift score > 1.0 trigger HIGH severity alerts.

**Kolmogorov-Smirnov Test:**

Applied two-sample KS test to compare empirical CDFs:

$$D_{n,m} = \sup_x |F_n(x) - G_m(x)|$$

Where $F_n$ and $G_m$ are empirical distribution functions of reference and current data.

Null hypothesis: distributions are identical. Reject if p-value < 0.05.

#### 4.3 Alert System

**Three-tier severity classification:**

1. **CRITICAL** (p-value < 0.01)
   - Recall drops below 50%
   - Immediate model retraining required

2. **HIGH** (0.01 < p-value < 0.05)
   - F1-score decreases by >20% from baseline
   - >30% of features exhibit significant drift

3. **MEDIUM** (0.05 < p-value < 0.10)
   - Prediction distribution shifts by >10%
   - Requires investigation within 48 hours

#### 4.4 Simulation Study

Simulated production data with controlled drift levels (0%, 10%, 20%, 40%) to validate monitoring sensitivity:

- **Week 1:** No alerts (drift = 0%)
- **Week 2:** 1 medium alert (drift = 10%)
- **Week 3:** 2 high alerts (drift = 20%)
- **Week 4:** 3 alerts including 1 critical (drift = 40%)

Demonstrates effective early detection of model degradation.

---

## Key Results Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Final ROC-AUC | 0.6475 | Moderate discrimination; typical for credit risk |
| Recall (High Risk) | 0.6889 | Captures 69% of actual defaults |
| Optimal Threshold | 0.08 | Aggressive risk detection strategy |
| Features Selected | 30 | Reduced from 47 post-engineering |
| Drift Detection Rate | 100% | Detected drift at 20%+ severity |

---

## Technical Contributions

1. **Multi-table feature aggregation** strategy preserving temporal patterns
2. **Consensus feature selection** combining information theory and ensemble importance
3. **Threshold optimization** framework for imbalanced classification
4. **Production-grade monitoring** with statistical drift detection and automated alerting
5. **Interpretability integration** via SHAP analysis for regulatory compliance

---

## Limitations & Future Work

**Current Limitations:**
- Limited feature set constrains predictive ceiling
- No temporal validation (time-series cross-validation)
- Threshold optimization on test set (potential overfitting)

**Proposed Enhancements:**
1. **Advanced drift detection:** Implement Maximum Mean Discrepancy (MMD) for multivariate drift
2. **Online learning:** Incremental model updates with streaming data
3. **Causal inference:** Distinguish correlation from causation in feature effects
4. **Fairness metrics:** Audit for demographic parity and equalized odds
5. **Ensemble methods:** Stacking/blending to improve stability

---

## Repository Structure

```
credit-risk-mlops/
├── notebooks/
│   ├── 01_EDA_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_evaluation.ipynb
│   └── 04_mlops_monitoring.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── best_model_lgbm.pkl
│   ├── model_config.pkl
│   └── preprocessing_pipeline.pkl
├── monitoring/
│   ├── monitoring_data.pkl
│   └── data_drift_report.csv
└── README.md
```

---

## References

**Methodological:**
- Chawla et al. (2002) - SMOTE: Synthetic Minority Over-sampling Technique
- Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions (SHAP)
- Chen & Guestrin (2016) - XGBoost: A Scalable Tree Boosting System
- Ke et al. (2017) - LightGBM: A Highly Efficient Gradient Boosting Decision Tree

**Domain-Specific:**
- Basel Committee on Banking Supervision - Guidelines on Credit Risk Management
- European Banking Authority (EBA) - Model Risk Management Framework

---

## Author

Developed as part of an application for the Machine Learning Intern position at Qonto, focusing on quantitative risk modeling and MLOps best practices in regulated fintech environments.