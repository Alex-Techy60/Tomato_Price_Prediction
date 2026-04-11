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

print("\n🚀 Comparing ALL Models (Statistical + ML + Baselines)...\n")

# Feature engineering for ML models
def create_features(data):
    temp = data.copy()
    temp['lag_1'] = temp['Price'].shift(1)
    temp['lag_2'] = temp['Price'].shift(2)
    temp['lag_3'] = temp['Price'].shift(3)
    temp['month'] = temp.index.month
    temp['year'] = temp.index.year
    return temp.dropna()

for state in states:
    print(f"📍 State: {state}")

    try:
        state_df = df[df['State'] == state].copy()
        state_df = state_df.set_index('Date')
        state_df = state_df[['Price']]
        state_df = state_df.resample('ME').mean().dropna()

        split_date = '2025-01-01'

        # =========================
        # Train-Test Split
        # =========================
        train = state_df[state_df.index < split_date]
        test = state_df[state_df.index >= split_date]

        if len(test) == 0 or len(train) < 12:
            print("⚠️ Not enough data, skipping...\n")
            continue

        # =========================
        # 🔵 SARIMA
        # =========================
        sarima_model = SARIMAX(
            train['Price'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        sarima_fit = sarima_model.fit(disp=False)
        sarima_pred = sarima_fit.forecast(steps=len(test))
        sarima_mae = mean_absolute_error(test['Price'], sarima_pred)

        # =========================
        # 🔴 PROPHET
        # =========================
        train_prophet = train.reset_index()
        train_prophet.columns = ['ds', 'y']

        prophet_model = Prophet()
        prophet_model.fit(train_prophet)

        future = prophet_model.make_future_dataframe(periods=len(test), freq='M')
        forecast = prophet_model.predict(future)

        prophet_pred = forecast.set_index('ds')['yhat'][-len(test):]
        prophet_mae = mean_absolute_error(test['Price'], prophet_pred)

        # =========================
        # 🟢 XGBOOST + 🟡 RANDOM FOREST
        # =========================
        ml_df = create_features(state_df)
        train_ml = ml_df[ml_df.index < split_date]
        test_ml = ml_df[ml_df.index >= split_date]

        X_train = train_ml.drop(columns=['Price'])
        y_train = train_ml['Price']
        X_test = test_ml.drop(columns=['Price'])
        y_test = test_ml['Price']

        xgb_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)

        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)

        # =========================
        # ⚪ BASELINE MODELS
        # =========================
        # Mean
        mean_value = train['Price'].mean()
        mean_pred = pd.Series([mean_value] * len(test), index=test.index)
        mean_mae = mean_absolute_error(test['Price'], mean_pred)

        # Naive
        naive_value = train['Price'].iloc[-1]
        naive_pred = pd.Series([naive_value] * len(test), index=test.index)
        naive_mae = mean_absolute_error(test['Price'], naive_pred)

        # Seasonal Naive
        seasonal_period = 12
        seasonal_values = train['Price'].iloc[-seasonal_period:].values
        seasonal_pred = pd.Series(
            [seasonal_values[i % seasonal_period] for i in range(len(test))],
            index=test.index
        )
        seasonal_mae = mean_absolute_error(test['Price'], seasonal_pred)

        # Drift
        first_value = train['Price'].iloc[0]
        last_value = train['Price'].iloc[-1]
        n = len(train)

        drift_pred = pd.Series(
            [last_value + h * ((last_value - first_value) / (n - 1)) for h in range(1, len(test) + 1)],
            index=test.index
        )
        drift_mae = mean_absolute_error(test['Price'], drift_pred)

        # =========================
        # 📈 COMBINED GRAPH
        # =========================
        plt.figure(figsize=(14, 7))

        plt.plot(state_df.index, state_df['Price'], label='Actual', linewidth=2)
        plt.plot(test.index, sarima_pred, label=f'SARIMA ({sarima_mae:.1f})')
        plt.plot(test.index, prophet_pred, label=f'Prophet ({prophet_mae:.1f})')
        plt.plot(test_ml.index, xgb_pred, label=f'XGBoost ({xgb_mae:.1f})')
        plt.plot(test_ml.index, rf_pred, label=f'RandomForest ({rf_mae:.1f})')
        plt.plot(test.index, mean_pred, '--', label=f'Mean ({mean_mae:.1f})')
        plt.plot(test.index, naive_pred, '--', label=f'Naive ({naive_mae:.1f})')
        plt.plot(test.index, seasonal_pred, '--', label=f'Seasonal Naive ({seasonal_mae:.1f})')
        plt.plot(test.index, drift_pred, '--', label=f'Drift ({drift_mae:.1f})')

        plt.axvline(pd.to_datetime(split_date), linestyle='--', color='black', label='Train/Test Split')

        plt.title(f"{state} Model Comparison (All Models)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(fontsize=8)
        plt.tight_layout()

        plt.savefig(f"comparison_model/{state}_all_models_with_baselines.png")
        plt.close()

        print(f"📊 Saved: {state}_all_models_with_baselines.png")
        print(f"   SARIMA: {sarima_mae:.2f}")
        print(f"   Prophet: {prophet_mae:.2f}")
        print(f"   XGBoost: {xgb_mae:.2f}")
        print(f"   RandomForest: {rf_mae:.2f}")
        print(f"   Mean: {mean_mae:.2f}")
        print(f"   Naive: {naive_mae:.2f}")
        print(f"   Seasonal Naive: {seasonal_mae:.2f}")
        print(f"   Drift: {drift_mae:.2f}\n")

    except Exception as e:
        print(f"❌ Error in {state}: {e}\n")

print("🎯 All Models Comparison Complete!")