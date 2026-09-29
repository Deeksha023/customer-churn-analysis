# Customer Churn Analysis and Prediction

A beginner-friendly, end-to-end Machine Learning project to analyze customer behavior and predict whether a customer will churn or not.

---

## 🎯 Objective
The primary objective of this project is to build and evaluate classification machine learning models to predict **Customer Churn** (`Yes` or `No`). By identifying at-risk customers early, subscription businesses (such as telecommunications and internet providers) can deploy targeted retention strategies, customer support, and tailored discounts to reduce customer attrition.

---

## 📊 Dataset
- **Dataset Name**: Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Location**: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Total Records (Rows)**: 7,043 customers
- **Total Attributes (Columns)**: 21 (reduced to 20 after removing `customerID`)
- **Target Column**: `Churn`
  - `Yes` = Customer churned (1)
  - `No` = Customer remained with the company (0)
- **Class Balance**: 
  - Retained: 5,174 customers (73.46%)
  - Churned: 1,869 customers (26.54%)

---

## 💻 Technologies
- **Python**: Core programming language
- **Jupyter Notebook**: Interactive analysis and visualization environment
- **NumPy**: Numerical operations and array manipulation
- **Pandas**: Tabular data manipulation, cleaning, and preprocessing
- **Matplotlib**: Core plotting and dashboard visualizations
- **Seaborn**: Statistical charts (heatmaps, countplots, KDE, boxplots)
- **Scikit-learn**: Data preprocessing, feature scaling, model training, evaluation metrics, and clustering
- **Joblib**: Model serialization and persistence to disk

---

## 🤖 Algorithms Used

### 1. Logistic Regression
- **What is it?** A linear probabilistic classification model.
- **Why use it?** Fast, transparent, and outputs calibrated probabilities using the Sigmoid activation function.
- **How it works?** Computes $z = w^T x + b$ and passes it through $\sigma(z) = \frac{1}{1 + e^{-z}}$. If probability $\ge 0.5$, it predicts churn.
- **Problem Type**: Binary Classification.

### 2. Decision Tree Classifier
- **What is it?** A non-parametric supervised learning algorithm that creates hierarchical if-else decision rules.
- **Why use it?** Captures non-linear relationships, intuitive rule-based interpretability, invariant to feature scaling.
- **How it works?** Recursively splits features to minimize Gini Impurity ($1 - \sum p_i^2$). Constrained to `max_depth = 4` to prevent overfitting.
- **Problem Type**: Classification & Regression.

### 3. Random Forest Classifier
- **What is it?** An ensemble of 100 Decision Trees.
- **Why use it?** Drastically reduces the high variance and overfitting of single decision trees.
- **How it works?** Combines **Bagging** (Bootstrap Aggregating) with **Random Feature Subsets** ($\sqrt{p}$). Each tree votes and the majority vote decides the final prediction.
- **Problem Type**: Tabular Classification & Regression.

### 4. Optional: K-Means Clustering (Unsupervised Learning)
- **What is it?** An unsupervised clustering algorithm.
- **Why use it?** Groups customers into behavioral segments (`tenure`, `MonthlyCharges`, `TotalCharges`) without using target labels.
- **How it works?** Identifies optimal $k = 3$ using the **Elbow Method** and iteratively assigns points to nearest cluster centroids.

---

## 🔄 Project Workflow
The Jupyter Notebook is organized into **27 clear, beginner-friendly sections**:

1. **Import Libraries**: Load essential Python packages.
2. **Load Dataset**: Read the CSV dataset into a Pandas DataFrame.
3. **Understand Dataset**: Inspect `.shape`, `.info()`, and `.describe()`.
4. **Data Cleaning**: Check duplicates (0 found), convert `TotalCharges` from string to float, fill 11 zero-tenure records with `0.0`, drop `customerID`.
5. **Missing Values**: Verify that 0 missing values remain across all columns.
6. **Exploratory Data Analysis (EDA)**: Calculate churn counts and percentages.
7. **Data Visualization**: 6 clear plots (Churn distribution, Churn by gender, Churn by contract, Churn by internet service, Churn by tenure, Churn by monthly charges).
8. **Correlation Heatmap**: Pearson correlation between numeric features and churn.
9. **Encode Categorical Data**: One-hot encode features using `pd.get_dummies(..., drop_first=True, dtype=int)` and map `Churn` to 0/1.
10. **Define X and y**: Separate features ($X$) and target ($y$).
11. **Train-Test Split**: Stratified 80% train and 20% test split (`test_size=0.20, random_state=42, stratify=y`).
12. **Feature Scaling**: Standardize numerical features using `StandardScaler` (fit strictly on train).
13. **Logistic Regression**: Train, predict, and evaluate model 1.
14. **Decision Tree**: Train, predict, and evaluate model 2 with `max_depth = 4`.
15. **Random Forest**: Train, predict, and evaluate model 3 with 100 trees.
16. **Model Predictions**: Side-by-side table comparing actual vs predicted labels for test samples.
17. **Confusion Matrix**: Annotated heatmaps showing TP, TN, FP, FN for all models.
18. **Accuracy, Precision, Recall and F1 Score**: Define and compute the 4 core metrics.
19. **Classification Report**: Full per-class precision, recall, and f1-score reports.
20. **Model Comparison**: Consolidated comparison DataFrame and grouped bar chart.
21. **Feature Importance**: Random Forest Gini importances bar chart highlighting top churn drivers.
22. **Overfitting and Underfitting**: Decision Tree `max_depth` (1 to 15) train vs test curve demonstration.
23. **Gradient Descent Explanation**: Beginner analogy + relation to Logistic Regression + mathematical demonstration.
24. **Save and Load Models**: Serialize models with `joblib.dump()` and run real-time inference with `joblib.load()`.
25. **Optional K-Means Customer Segmentation**: Elbow curve and 3 customer personas.
26. **ML Concepts Used**: Complete mapping table covering all 26+ concepts.
27. **Final Conclusion**: Technical summary and business recommendations.

---

## 📈 Results & Model Comparison

Evaluated on the unseen test set of **1,409 customers**:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** | **80.41%** | 65.64% | **53.74%** | **59.10%** |
| **Decision Tree (depth=4)** | 79.56% | **66.81%** | 41.71% | 51.35% |
| **Random Forest (100 trees)** | **80.06%** | 65.73% | 50.53% | **57.14%** |

### Top Churn Drivers (Feature Importance):
1. **`tenure` (20.3%)**: Short-tenure customers in their first 1-10 months churn at the highest rates.
2. **`TotalCharges` (14.5%) & `MonthlyCharges` (9.4%)**: High monthly bills strongly incentivize customers to leave.
3. **`InternetService_Fiber optic` (9.7%)**: Fiber customers experience higher churn rates.
4. **`Contract_Two year` (8.2%)**: Multi-year contracts are the strongest protective factor against churn.
5. **`PaymentMethod_Electronic check` (6.0%)**: Associated with higher churn compared to automatic payment methods.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open and run all cells in [`Customer_Churn_Analysis.ipynb`](file:///c:/customer_churn_analysis/Customer_Churn_Analysis.ipynb):
```bash
jupyter notebook Customer_Churn_Analysis.ipynb
```
*(All 27 sections are pre-computed with pre-rendered plots and tables!)*

### 3. Run Real-Time CLI Inference
Test the trained model on sample customer profiles from the command line:
```bash
python predict.py
```

---

## 💡 Conclusion & Business Takeaways

1. **Incentivize 1-Year and 2-Year Contracts**: Month-to-month contracts have the highest churn rate (~43%). Offering a 10-15% discount for annual commitments will directly stabilize revenue.
2. **First-Year Onboarding Focus**: Churn is heavily concentrated in the first 10 months. Introducing proactive onboarding check-ins during the first 90 days will reduce customer attrition.
3. **Bundle Value-Added Services**: Customers with `OnlineSecurity` and `TechSupport` have much higher retention. Bundling these into standard packages increases customer stickiness.
4. **Targeted Campaigns on High-Spend At-Risk Customers**: Customer segmentation (Cluster 1) identifies high-spending, short-tenure customers who need proactive loyalty discounts before they switch to competitors.
