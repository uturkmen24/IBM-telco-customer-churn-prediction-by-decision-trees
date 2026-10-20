# 📡 Telco Customer Churn Prediction by Decision Trees

**Author:** Umut Türkmen  

## 📌 1. Project Overview & Business Problem
In the telecommunications industry, acquiring a new customer is significantly more expensive than retaining an existing one. Therefore, predicting **Customer Churn** (identifying which customers are likely to cancel their subscriptions) is a critical business challenge.

This end-to-end data science project analyzes customer demographics, account details, and service subscriptions using the **IBM Telco Customer Churn Dataset** to build a binary classification model. Beyond achieving high predictive accuracy, the primary goal is to provide **explainable business rules** that help decision-makers take proactive retention actions (such as targeted contract upgrades or discount campaigns).

---

## 📊 2. Dataset & Data Cleaning
The dataset (`IBM-Telco-Customer-Churn.csv`) contains **7,043 rows** and **21 features** representing customer profiles, billing metrics, and service add-ons, with `Churn` (`Yes` / `No`) as the target variable.

### Key Data Preprocessing Steps:
* **Uncovering Hidden Missing Values (`TotalCharges`):** Initially, `isnull().sum()` reported zero missing values, yet the `TotalCharges` column was stored as an `object` (string) instead of `float64`. Filtering the dataset revealed **11 newly joined customers (`tenure == 0`)** whose `TotalCharges` contained empty whitespace strings (`" "`). These were coerced into numerical format and filled with `0.0`, as these customers had not yet completed their first billing cycle.
* **Dropping Identifiers:** The `customerID` column was removed since unique random identifiers carry no predictive signal and lead to overfitting.
* **Evidence-Based Feature Selection:** Based on Exploratory Data Analysis (EDA) and tree importance checks, noisy variables with negligible impact on churn (e.g., `gender`, where churn rates were nearly identical at 26.9% vs. 26.1%) were excluded. We selected **8 high-impact features** covering financial, service, and demographic dimensions:
  * **Numerical (3):** `tenure`, `MonthlyCharges`, `TotalCharges`
  * **Categorical / Binary (5):** `Contract`, `InternetService`, `TechSupport`, `OnlineSecurity`, `SeniorCitizen`

---

## 🔍 3. Exploratory Data Analysis (EDA) & Key Insights
Through statistical grouping (`groupby`, `crosstab`) and `Matplotlib` visualizations, several critical patterns emerged:

1. **Class Imbalance:** Approximately **73.5%** of customers stayed (`No`), while **26.5%** churned (`Yes`).
2. **The Power of Contract Type:** Customers on **Month-to-month** contracts had an alarming churn rate of **42.7%**, compared to **11.3%** for one-year contracts and just **2.8%** for two-year contracts.
3. **Internet Service Vulnerability:** Users with **Fiber optic** internet exhibited a **41.9%** churn rate, significantly higher than DSL users (**18.9%**) or customers without internet service (**7.4%**).
4. **Tenure vs. Monthly Charges Interaction:** A scatter plot analysis demonstrated that churned customers heavily cluster in the **low-tenure (< 20 months) and high-monthly-charge (> $70)** region. On average, churners stayed for ~18 months and paid ~$74.4/month, whereas retained customers stayed for ~37.6 months and paid ~$61.3/month.

---

## 🧠 4. Model Selection & Training
The dataset was split into **80% training (5,634 samples)** and **20% testing (1,409 samples)** sets (`random_state=42`). Categorical variables were mapped into numerical representations (`0, 1, 2`) preserving ordinal relationships where applicable (such as contract duration).

### Why Decision Tree Classifier?
We selected a **Decision Tree Classifier (`max_depth=4`)** for three strategic reasons:
1. **Handling Mixed Data Types:** It naturally processes both continuous financial metrics (`MonthlyCharges`) and encoded categorical variables (`Contract`) without requiring feature scaling.
2. **Capturing Non-Linear Interactions:** Customer churn is driven by conditional combinations (e.g., *high monthly charges AND month-to-month contract*), which decision trees capture via hierarchical "if-then" splits.
3. **Explainable AI ("White-Box" Model):** Unlike black-box algorithms, visualizing the decision tree (`plot_tree`) allows telecom stakeholders to clearly see the exact thresholds (`Contract <= 0.5`, `InternetService <= 1.5`, `tenure <= 14.5`) driving customer loss.

---

## 📈 5. Evaluation, Results & Limitations
The trained model was evaluated on both training and unseen test sets to verify generalization:

* **Training Accuracy:** `79.39%`
* **Test Accuracy:** `79.70%`

> **Zero Overfitting:** The nearly identical training and test accuracy scores prove that constraining the tree with `max_depth=4` successfully prevented memorization (overfitting), allowing the model to learn generalizable patterns.

### Confusion Matrix & Classification Report (Test Set)
| Metric / Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Stayed (`0`)** | 0.83 | 0.91 | 0.87 | 1,036 |
| **Churned (`1`)** | 0.66 | 0.49 | 0.56 | 373 |
| **Overall Accuracy** | - | - | **0.80** | **1,409** |

* **True Negatives:** 941 | **False Positives:** 95  
* **False Negatives:** 191 | **True Positives:** 182  

### Critical Discussion & Future Improvements
1. **Addressing Class Imbalance (Recall Optimization):** Because 73.5% of the dataset consists of retained customers, the model excels at identifying staying customers (`Recall = 0.91`) but misses some actual churners (`Recall = 0.49`). In future iterations, applying `class_weight='balanced'` or SMOTE oversampling will be used to prioritize catching high-risk churners.
2. **Preventing Micro-Splits:** Visualizing the tree revealed two bottom-right leaf nodes containing only 5 and 6 samples (high-paying customers on long-term contracts). Setting `min_samples_leaf=20` or transitioning to ensemble algorithms like **Random Forest** or **XGBoost** will eliminate statistically unreliable micro-leaves.

---

## 💻 6. Smart Prediction Tool (Mini Application)
To bridge the gap between model training and real-world deployment, the notebook includes an interactive prediction function: `predict_churn()`. 
* It accepts 8 customer parameters via user `input()`, converts them into a structured `DataFrame`, and utilizes both `model.predict()` and `model.predict_proba()`.
* Instead of returning a raw `0` or `1`, it outputs an actionable business recommendation with the exact probability percentage (e.g., *"There is a 78.40% probability that the customer belongs to the churn group. The company needs to urgently offer a commitment/discount campaign to this customer."*).

---

## 📁 7. Repository Structure
```text
├── telco_customer_churn_prediction_model.ipynb  # End-to-end Jupyter Notebook (EDA, Model, App)
├── IBM-Telco-Customer-Churn.csv                 # Dataset (7,043 rows, 21 columns)
├── Project_Report.pdf                           # 1-Page Executive Summary & Findings Report
├── Project_Presentation.pdf                     # Project Presentation Slides
└── README.md                                    # Project Documentation
```

---

## ⚙️ 8. How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/telco-customer-churn-prediction-by-decision-trees.git
   cd telco-customer-churn-prediction-by-decision-trees
   ```
2. Install the required Python libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
3. Open `telco_customer_churn_prediction_model.ipynb` in **VS Code** or **Jupyter Lab/Notebook** and run all cells sequentially.
