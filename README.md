# Customer Churn Analysis and Prediction
### An End-to-End Interactive Machine Learning Project & Web Application

A professional, beginner-friendly Machine Learning application to analyze customer behavior, visualize dataset patterns, and predict in real-time whether a subscriber is likely to churn or remain with the company.

---

## 🎯 Objective

The primary objective of this project is to build, evaluate, and deploy classification machine learning models that predict **Customer Churn** (`Yes` or `No`). By proactively identifying at-risk customers, subscription businesses (such as telecommunications, internet, and SaaS providers) can deploy targeted retention strategies, customer support, and tailored renewal incentives to reduce churn and protect recurring revenue.

---

## ✨ Features

- **Interactive Modern Web Dashboard**: Built with **FastAPI**, **Vanilla CSS**, and **Chart.js**, featuring dark mode, glassmorphism, responsive cards, and dynamic visual indicators.
- **Real-Time ML Inference**: Instant scoring using the pre-trained **Random Forest Classifier** (`models/random_forest_model.joblib`) with probability gauges and risk badges (Low, Medium, High).
- **Configurable Risk Thresholds**:
  - `0% - 30%`: **Low Risk** (Customer likely retained)
  - `30% - 60%`: **Medium Risk** (Monitoring & attention recommended)
  - `60% - 100%`: **High Risk** (Customer may churn)
- **Dataset Visualizations & Analytics**: Interactive charts rendered directly from the 7,043 customer records in the dataset:
  1. Churn by Contract Type
  2. Churn by Internet Service
  3. Churn by Tenure Cohort
  4. Churn by Payment Method
- **Customer Risk Explanations**: Evidence-based factors derived from the customer's specific attributes and historical project findings.
- **Actionable Business Recommendations**: Separated retention strategies tailored to the calculated risk tier.
- **Prediction History Log**: Local session table tracking past predictions with timestamp, probability, risk level, and a "Clear History" button.
- **1-Click Profile Presets**: Test high-risk and low-risk customer profiles with a single click.
- **Dual Inference Modes**: Run either via the interactive web dashboard or the command-line script (`python predict.py`).

---

## 💻 Technologies Used

| Technology | Category | Purpose |
| :--- | :--- | :--- |
| **Python 3.10+** | Programming Language | Core backend and machine learning pipelines |
| **FastAPI** | Web Framework | Lightweight, high-performance async REST API and file server |
| **Uvicorn** | ASGI Server | Production-ready web server |
| **Scikit-learn** | Machine Learning | Feature preprocessing, scaling, and classification modeling |
| **Pandas & NumPy** | Data Analysis | Data manipulation, cleaning, and tabular encoding |
| **Joblib** | Model Persistence | Serialization and loading of trained models and metadata |
| **HTML5 & Vanilla CSS** | Frontend | Responsive dashboard, glassmorphic cards, modern typography |
| **Vanilla JavaScript** | Frontend Logic | API communication, form handling, and local storage |
| **Chart.js** | Data Visualization | Interactive animated charts for exploratory dataset analytics |
| **Matplotlib & Seaborn** | Exploratory Analysis | Offline visualization and statistical heatmaps in notebook |

---

## 📊 Dataset

- **Dataset Name**: Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Location**: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Total Records (Rows)**: 7,043 customers
- **Total Attributes (Columns)**: 21 (reduced to 20 after dropping arbitrary `customerID`)
- **Target Column**: `Churn`
  - `Yes` = Customer churned (1) — 1,869 customers (26.54%)
  - `No` = Customer retained (0) — 5,174 customers (73.46%)

---

## 🤖 Machine Learning Algorithms

1. **Random Forest Classifier (Active Production Model)**:
   - **Type**: Ensemble of 100 bootstrapped Decision Trees (`RandomForestClassifier(n_estimators=100)`).
   - **Why use it?** Combines bagging and random feature subsets to drastically reduce variance and prevent overfitting.
   - **Performance**: **80.55% Accuracy**, 0.66 Precision, 0.54 Recall, 0.59 F1-score.
   - **Persistence**: Saved in `models/random_forest_model.joblib`.

2. **Logistic Regression (Benchmarked Baseline)**:
   - **Type**: Linear probabilistic classification using Sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$.
   - **Performance**: **80.34% Accuracy**, 0.65 Precision, 0.54 Recall, 0.59 F1-score.
   - **Persistence**: Saved in `models/logistic_regression_model.joblib`.

3. **Decision Tree Classifier (Benchmarked Rule-Based Model)**:
   - **Type**: Non-parametric tree splitting constrained to `max_depth = 4` to prevent memorization.
   - **Performance**: **79.13% Accuracy**, 0.62 Precision, 0.51 Recall, 0.56 F1-score.

---

## 📁 Project Structure

```text
customer_churn_analysis/
│
├── app.py                            # FastAPI backend application & API endpoints
├── predict.py                        # Standalone CLI inference script
├── requirements.txt                  # Python dependencies
├── build_full_notebook.py            # Automated notebook generation script
├── Customer_Churn_Analysis.ipynb     # Complete 27-section Jupyter Notebook
├── README.md                         # Project documentation
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Historical dataset (7,043 rows)
│
├── models/
│   ├── random_forest_model.joblib    # Trained Random Forest classifier (100 trees)
│   ├── logistic_regression_model.joblib # Trained Logistic Regression model
│   ├── model_columns.joblib          # List of 30 exact one-hot encoded feature names
│   └── scaler.joblib                 # StandardScaler fitted on numerical features
│
└── static/
    ├── index.html                    # Single-page web dashboard
    ├── style.css                     # Modern dark-mode glassmorphic styling
    └── app.js                        # Client-side validation, Chart.js, & API integration
```

---

## 🚀 Installation & Setup

### 1. Clone or Open the Repository
```bash
cd c:\customer_churn_analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

### Option A: Launch the Web Dashboard (Recommended)

Start the web application server:
```bash
python app.py
```
*Alternatively, start using Uvicorn directly:*
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

### Option B: Run Command-Line Inference (CLI)

Test the pre-trained model directly from your terminal:
```bash
python predict.py
```

**Sample Output:**
```text
=================================================================
      CUSTOMER CHURN PREDICTION - REAL-TIME INFERENCE DEMO      
=================================================================

[1] Loading trained model artifacts from 'models/'...
    -> Successfully loaded Random Forest Classifier.
    -> Total input features expected: 30

[2] Evaluating Sample Customer Profiles:
-----------------------------------------------------------------
PROFILE A: New Customer on Month-to-Month Plan ($95.00/mo, Fiber Optic)
  -> Prediction:        Yes (Churn)
  -> Churn Probability: 68.17%
  -> Risk Level:        High
-----------------------------------------------------------------
PROFILE B: Loyal Customer on 2-Year Contract ($45.00/mo, Tech Support)
  -> Prediction:        No (Retained)
  -> Churn Probability: 3.14%
  -> Risk Level:        Low
-----------------------------------------------------------------

Inference successfully demonstrated!
```

---

### Option C: Run the Jupyter Notebook

Open and inspect the full 27-section training pipeline, visualizations, and clustering:
```bash
jupyter notebook Customer_Churn_Analysis.ipynb
```

---

## 📖 How to Use the Web Application

1. **Review Dashboard KPIs**: Look at the top summary cards showing Total Customers (7,043), Retained (5,174), Churned (1,869), and the Overall Churn Rate (26.54%).
2. **Enter Customer Information**:
   - Use the dropdowns, range slider, and numerical inputs to configure customer attributes.
   - Or click **"⚡ Load High-Risk Sample"** or **"🛡️ Load Low-Risk Sample"** for quick 1-click testing.
3. **Click "PREDICT CHURN"**:
   - The application securely sends the payload to `/api/predict`.
   - The backend encodes the attributes into the exact 30 features expected by `models/model_columns.joblib`.
   - The Random Forest model computes the churn probability.
4. **Inspect the Result Card**:
   - **Prediction Banner**: Displays either `HIGH RISK - CUSTOMER MAY CHURN`, `MEDIUM RISK`, or `LOW RISK - CUSTOMER LIKELY RETAINED`.
   - **Probability Bar**: Shows exact probability percentage.
   - **Why this customer is at risk**: Detailed evidence points explaining the prediction.
   - **Recommended Actions**: Clear business retention actions tailored to the customer.
5. **View Prediction History**: Scroll to the Prediction History table to compare multiple evaluations across your session.
6. **Explore Analytics**: Check the interactive charts to see how contracts, payment methods, internet services, and tenure impact churn across the whole dataset.

---

## 🔍 Example Predictions

### Profile A: At-Risk Customer
- **Attributes**: Month-to-month contract, tenure = 2 months, Monthly Charges = $95.00, Fiber Optic, Electronic Check, no Tech Support.
- **Model Output**:
  - **Prediction**: `HIGH RISK - CUSTOMER MAY CHURN`
  - **Probability**: `68.17%` (High Risk)
  - **Recommended Action**: Offer a 15-20% discount on an annual contract and provide a free trial of Premium Tech Support.

### Profile B: Loyal Customer
- **Attributes**: Two-year contract, tenure = 60 months, Monthly Charges = $45.00, DSL, Credit Card Auto-pay, Tech Support and Online Security active.
- **Model Output**:
  - **Prediction**: `LOW RISK - CUSTOMER LIKELY RETAINED`
  - **Probability**: `3.14%` (Low Risk)
  - **Recommended Action**: Enroll in VIP loyalty rewards and offer early access to service upgrades.

---

## 🔮 Future Enhancements

1. **Customer Database Integration**: Connect to PostgreSQL or MongoDB to automatically score customer profiles on a scheduled daily batch job.
2. **Explainability with SHAP / LIME**: Add interactive waterfall charts showing individual feature contributions per customer.
3. **Automated Retention Triggers**: Webhook integration with CRM platforms (e.g., HubSpot, Salesforce) to auto-dispatch retention emails or discount codes when probability exceeds 60%.
4. **Hyperparameter Tuning**: Explore XGBoost, LightGBM, and CatBoost with Bayesian optimization for additional accuracy gains.
