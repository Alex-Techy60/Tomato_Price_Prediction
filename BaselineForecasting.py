import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error

# 🔥 Create directory
os.makedirs("baseline_models", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("cleaned_tomato_prices.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Ensure Price is numeric
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop bad rows
df = df.dropna()

# Get unique states
states = df['State'].unique()

print("\n🚀 Starting Baseline Forecasting Models...\n")

for state in states:
    print(f"📍 State: {state}")

    try:
        # Filter state data
        state_df = df[df['State'] == state].copy()
        state_df = state_df.set_index('Date')
        state_df = state_df[['Price']]

        # Monthly resampling
        state_df = state_df.resample('ME').mean().dropna()

        # Train-test split
        split_date = '2025-01-01'
        train = state_df[state_df.index < split_date]
        test = state_df[state_df.index >= split_date]

        if len(test) == 0 or len(train) < 12:
            print("⚠️ Not enough data, skipping...\n")
            continue

        # =========================
        # 1️⃣ Mean Method
        # =========================
        mean_value = train['Price'].mean()
        mean_pred = pd.Series([mean_value] * len(test), index=test.index)
        mean_mae = mean_absolute_error(test['Price'], mean_pred)

        # =========================
        # 2️⃣ Naive Method
        # =========================
        naive_value = train['Price'].iloc[-1]
        naive_pred = pd.Series([naive_value] * len(test), index=test.index)
        naive_mae = mean_absolute_error(test['Price'], naive_pred)

        # =========================
        # 3️⃣ Seasonal Naive Method
        # =========================
        seasonal_period = 12
        seasonal_values = train['Price'].iloc[-seasonal_period:].values
        seasonal_pred = []

        for i in range(len(test)):
            seasonal_pred.append(seasonal_values[i % seasonal_period])

        seasonal_pred = pd.Series(seasonal_pred, index=test.index)
        seasonal_mae = mean_absolute_error(test['Price'], seasonal_pred)

        # =========================
        # 4️⃣ Drift Method
        # =========================
        first_value = train['Price'].iloc[0]
        last_value = train['Price'].iloc[-1]
        n = len(train)

        drift_pred = []
        for h in range(1, len(test) + 1):
            forecast = last_value + h * ((last_value - first_value) / (n - 1))
            drift_pred.append(forecast)

        drift_pred = pd.Series(drift_pred, index=test.index)
        drift_mae = mean_absolute_error(test['Price'], drift_pred)

        # =========================
        # 📈 Combined Graph
        # =========================
        plt.figure(figsize=(12, 6))

        plt.plot(state_df.index, state_df['Price'], label='Actual', linewidth=2)
        plt.plot(test.index, mean_pred, label=f'Mean ({mean_mae:.1f})')
        plt.plot(test.index, naive_pred, label=f'Naive ({naive_mae:.1f})')
        plt.plot(test.index, seasonal_pred, label=f'Seasonal Naive ({seasonal_mae:.1f})')
        plt.plot(test.index, drift_pred, label=f'Drift ({drift_mae:.1f})')

        plt.axvline(pd.to_datetime(split_date), linestyle='--', label='Train/Test Split')

        plt.title(f"{state} Baseline Model Comparison")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"baseline_models/{state}_baseline_comparison.png")
        plt.close()

        print(f"📊 Saved: baseline_models/{state}_baseline_comparison.png")
        print(f"   Mean MAE: {mean_mae:.2f}")
        print(f"   Naive MAE: {naive_mae:.2f}")
        print(f"   Seasonal Naive MAE: {seasonal_mae:.2f}")
        print(f"   Drift MAE: {drift_mae:.2f}\n")

    except Exception as e:
        print(f"❌ Error in {state}: {e}\n")

print("🎯 Baseline Forecasting Complete!")