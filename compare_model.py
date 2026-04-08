import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Create folder
os.makedirs("comparison_model", exist_ok=True)

# Load data
df = pd.read_csv("cleaned_tomato_prices.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna()

states = df['State'].unique()

print("\n🚀 Comparing ALL 4 Models...\n")

# 🔥 Feature engineering (for ML models)
def create_features(data):
    df = data.copy()
    df['lag_1'] = df['Price'].shift(1)
    df['lag_2'] = df['Price'].shift(2)
    df['lag_3'] = df['Price'].shift(3)
    df['month'] = df.index.month
    df['year'] = df.index.year
    return df.dropna()

for state in states:
    print(f"📍 State: {state}")

    state_df = df[df['State'] == state].copy()
    state_df = state_df.set_index('Date')
    state_df = state_df[['Price']]
    state_df = state_df.resample('ME').mean().dropna()

    split_date = '2025-01-01'

    # =========================
    # 🔵 SARIMA
    # =========================
    train_sarima = state_df[state_df.index < split_date]
    test_sarima = state_df[state_df.index >= split_date]

    sarima_model = SARIMAX(
        train_sarima['Price'],
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.forecast(steps=len(test_sarima))
    sarima_mae = mean_absolute_error(test_sarima['Price'], sarima_pred)

    # =========================
    # 🔴 PROPHET
    # =========================
    train_prophet = train_sarima.reset_index()
    train_prophet.columns = ['ds', 'y']

    prophet_model = Prophet()
    prophet_model.fit(train_prophet)

    future = prophet_model.make_future_dataframe(periods=len(test_sarima), freq='M')
    forecast = prophet_model.predict(future)

    prophet_pred = forecast.set_index('ds')['yhat'][-len(test_sarima):]
    prophet_mae = mean_absolute_error(test_sarima['Price'], prophet_pred)

    # =========================
    # 🟢 XGBOOST
    # =========================
    xgb_df = create_features(state_df)
    train_xgb = xgb_df[xgb_df.index < split_date]
    test_xgb = xgb_df[xgb_df.index >= split_date]

    X_train_xgb = train_xgb.drop(columns=['Price'])
    y_train_xgb = train_xgb['Price']
    X_test_xgb = test_xgb.drop(columns=['Price'])
    y_test_xgb = test_xgb['Price']

    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
    xgb_model.fit(X_train_xgb, y_train_xgb)

    xgb_pred = xgb_model.predict(X_test_xgb)
    xgb_mae = mean_absolute_error(y_test_xgb, xgb_pred)

    # =========================
    # 🟡 RANDOM FOREST
    # =========================
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train_xgb, y_train_xgb)

    rf_pred = rf_model.predict(X_test_xgb)
    rf_mae = mean_absolute_error(y_test_xgb, rf_pred)

    # =========================
    # 📈 COMBINED GRAPH
    # =========================
    plt.figure(figsize=(10,5))

    # Actual
    plt.plot(state_df.index, state_df['Price'], label='Actual', linewidth=2)

    # Predictions
    plt.plot(test_sarima.index, sarima_pred, label=f'SARIMA ({sarima_mae:.1f})')
    plt.plot(test_sarima.index, prophet_pred, label=f'Prophet ({prophet_mae:.1f})')
    plt.plot(test_xgb.index, xgb_pred, label=f'XGBoost ({xgb_mae:.1f})')
    plt.plot(test_xgb.index, rf_pred, label=f'RandomForest ({rf_mae:.1f})')

    plt.axvline(pd.to_datetime(split_date), linestyle='--', label='Train/Test Split')

    plt.title(f"{state} Model Comparison (All 4)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"comparison_model/{state}_all_models.png")
    plt.close()

    print(f"📊 Saved: {state}_all_models.png")
    print(f"   SARIMA: {sarima_mae:.2f}")
    print(f"   Prophet: {prophet_mae:.2f}")
    print(f"   XGBoost: {xgb_mae:.2f}")
    print(f"   RandomForest: {rf_mae:.2f}\n")

print("🎯 All Models Comparison Complete!")