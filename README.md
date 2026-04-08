# 🍅 Tomato Price Prediction (Time Series Forecasting)

This project focuses on predicting tomato price fluctuations during harvest seasons using multiple time series and machine learning models. It analyzes real-world agricultural data and compares different modeling approaches to determine the most effective one.

---

## 📌 Objective

To forecast tomato prices and analyze supply shock patterns during harvest seasons using:

- Statistical Models
- Machine Learning Models

---

## 📊 Dataset

- Source: **Agmarknet (Government of India)**
- Duration: **2021 – 2026 (6 years)**
- States Selected:
  - Karnataka (Koram Mandi)
  - Maharashtra (Nashik Mandi)
  - Delhi (Azadpur Mandi)

---

## 🧹 Data Preprocessing

- Raw CSV files were collected for each state.
- Data had **no missing values**, so no imputation was required.
- Cleaned and standardized into:


Date | State | Price


- Aggregated and sorted for modeling.

---

## 🧠 Model Development Flow

### 1️⃣ SARIMA (Baseline Model)
- Used as the initial statistical model.
- Required stationarity → applied differencing.
- Limitations:
  - Could not handle sudden price spikes.

---

### 2️⃣ Prophet
- Introduced for better trend & seasonality modeling.
- Improved over SARIMA.
- Still struggled with abrupt fluctuations.

---

### 3️⃣ XGBoost
- Shifted to machine learning approach.
- Used lag-based feature engineering.
- Captured non-linear patterns and spikes effectively.
- Performed best in highly volatile data.

---

### 4️⃣ Random Forest
- Implemented as another ensemble model.
- Provided stable and consistent predictions.
- Outperformed others in moderately stable regions.

---

## 📈 Model Comparison

All four models were compared using:

- **Mean Absolute Error (MAE)**
- Visual comparison graphs

### 🔥 Observations:

| State       | Best Model      | Reason |
|------------|----------------|--------|
| Delhi      | XGBoost        | Handles high volatility |
| Karnataka  | Random Forest  | Stable prediction |
| Maharashtra| Random Forest  | Smooth & consistent |

---

## 📊 Visualization

- Individual model graphs
- Combined comparison graphs (All 4 models in one)

---

## 🧠 Key Insights

- No single model works best for all datasets.
- Statistical models struggle with real-world noise.
- Machine learning models perform better for:
  - non-linear patterns
  - supply shocks

---

## ⚙️ Tech Stack

- Python
- Pandas
- Matplotlib
- Statsmodels (SARIMA)
- Prophet (Facebook)
- Scikit-learn
- XGBoost

---

## 🚀 How to Run

```bash
# Step 1: Clean data
python clean_data.py

# Step 2: Run models
python sarima_model.py
python prophet_model.py
python xgboost_model.py
python random_forest_model.py

# Step 3: Compare models
python compare_models.py
📌 Project Structure
Tomato_Price_Prediction/
│
├── Tomato_Data/
├── clean_data.py
├── sarima_model.py
├── prophet_model.py
├── xgboost_model.py
├── random_forest_model.py
├── compare_models.py
│
├── sarima_model/
├── prophet_model/
├── xgboost_model/
├── random_forest_model/
├── comparison_model/
│
└── cleaned_tomato_prices.csv
🎯 Conclusion
Machine Learning models outperformed statistical models.
XGBoost is best for volatile data.
Random Forest is best for stable data.
Model selection depends on data characteristics.

 Author

Sugyan Singh
BTech CSE, IIIT Dharwad

⭐ Future Improvements
Add LSTM / Deep Learning models
Include external factors (weather, demand)
Real-time price prediction system