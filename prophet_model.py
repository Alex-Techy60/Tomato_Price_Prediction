import pandas as pd
import matplotlib.pyplot as plt
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# 🔥 Create directory
os.makedirs("prophet_model", exist_ok=True)

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

print("\n🚀 Starting Prophet Model Training...\n")

for state in states:
    print(f"📍 State: {state}")

    state_df = df[df['State'] == state].copy()
    state_df = state_df.set_index('Date')
    state_df = state_df[['Price']]
    state_df = state_df.resample('ME').mean().dropna()

    split_date = '2025-01-01'
    train = state_df[state_df.index < split_date]
    test = state_df[state_df.index >= split_date]

    if len(test) == 0:
        print("⚠️ No test data for 2025, skipping...\n")
        continue

    try:
        train_prophet = train.reset_index()
        train_prophet.columns = ['ds', 'y']

        model = Prophet()
        model.fit(train_prophet)

        future = model.make_future_dataframe(periods=len(test), freq='M')
        forecast = model.predict(future)

        forecast_values = forecast.set_index('ds')['yhat'][-len(test):]

        mae = mean_absolute_error(test['Price'], forecast_values)
        print(f"✅ MAE: {mae:.2f}")

        # FULL GRAPH
        plt.figure(figsize=(10,5))
        plt.plot(state_df.index, state_df['Price'], label='Actual (Full)')
        plt.plot(test.index, forecast_values, label='Predicted')
        plt.axvline(pd.to_datetime(split_date), linestyle='--', label='Train/Test Split')

        plt.title(f"{state} Tomato Price Prediction (Prophet)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"prophet_model/{state}_prophet_full.png")
        plt.close()

        # ZOOM GRAPH
        plt.figure(figsize=(10,5))
        plt.plot(test.index, test['Price'], label='Actual')
        plt.plot(test.index, forecast_values, label='Predicted')

        plt.title(f"{state} Prophet Prediction (Zoomed)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"prophet_model/{state}_prophet_zoomed.png")
        plt.close()

        print(f"📊 Prophet graphs saved for {state}\n")

    except Exception as e:
        print(f"❌ Error in {state}: {e}\n")

print("🎯 Prophet Modeling Complete!")