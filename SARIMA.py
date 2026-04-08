import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
import itertools

# 🔥 Create directory
os.makedirs("sarima_model", exist_ok=True)

# Load data
df = pd.read_csv("cleaned_tomato_prices.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna()

states = df['State'].unique()

print("\n🚀 Starting TUNED SARIMA Model...\n")

# Parameter ranges (small but effective)
p = d = q = range(0, 2)
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(range(0,2), range(0,2), range(0,2)))]

for state in states:
    print(f"📍 State: {state}")

    state_df = df[df['State'] == state].copy()
    state_df = state_df.set_index('Date')
    state_df = state_df[['Price']]

    # Monthly
    state_df = state_df.resample('ME').mean().dropna()

    # 🔥 Log transform
    state_df['Price'] = np.log(state_df['Price'])

    # 🔍 ADF Test
    result = adfuller(state_df['Price'])
    print(f"ADF p-value: {result[1]:.4f}")

    split_date = '2025-01-01'
    train = state_df[state_df.index < split_date]
    test = state_df[state_df.index >= split_date]

    if len(test) == 0:
        print("⚠️ No test data, skipping\n")
        continue

    best_aic = float("inf")
    best_order = None
    best_seasonal = None
    best_model = None

    print("🔎 Searching best parameters...")

    # 🔥 Grid Search
    for order in itertools.product(p, d, q):
        for seasonal in seasonal_pdq:
            try:
                model = SARIMAX(
                    train['Price'],
                    order=order,
                    seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                results = model.fit(disp=False)

                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
                    best_seasonal = seasonal
                    best_model = results

            except:
                continue

    print(f"✅ Best SARIMA{best_order} x {best_seasonal} - AIC:{best_aic:.2f}")

    # Forecast
    forecast_log = best_model.forecast(steps=len(test))

    # Convert back
    forecast = np.exp(forecast_log)
    test_actual = np.exp(test['Price'])

    # MAE
    mae = mean_absolute_error(test_actual, forecast)
    print(f"📊 MAE: {mae:.2f}")

    # =========================
    # 📈 FULL GRAPH
    # =========================
    plt.figure(figsize=(10,5))

    plt.plot(state_df.index, np.exp(state_df['Price']), label='Actual (Full)')
    plt.plot(test.index, forecast, label='Predicted')

    plt.axvline(pd.to_datetime(split_date), linestyle='--', label='Train/Test Split')

    plt.title(f"{state} Tuned SARIMA Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"sarima_model/{state}_tuned_full.png")
    plt.close()

    # =========================
    # 🔍 ZOOM GRAPH
    # =========================
    plt.figure(figsize=(10,5))

    plt.plot(test.index, test_actual, label='Actual')
    plt.plot(test.index, forecast, label='Predicted')

    plt.title(f"{state} Tuned SARIMA (Zoomed)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"sarima_model/{state}_tuned_zoom.png")
    plt.close()

    print(f"📊 Graphs saved for {state}\n")

print("🎯 Tuned SARIMA Complete!")