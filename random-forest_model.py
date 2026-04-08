import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 🔥 Create directory
os.makedirs("random_forest_model", exist_ok=True)

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

print("\n🚀 Starting Random Forest Model...\n")

# 🔥 Feature engineering function
def create_features(data):
    df = data.copy()
    
    # Lag features
    df['lag_1'] = df['Price'].shift(1)
    df['lag_2'] = df['Price'].shift(2)
    df['lag_3'] = df['Price'].shift(3)
    
    # Time features
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    return df.dropna()

for state in states:
    print(f"📍 State: {state}")

    # Filter state data
    state_df = df[df['State'] == state].copy()
    state_df = state_df.set_index('Date')

    # Keep only price
    state_df = state_df[['Price']]

    # Monthly resampling
    state_df = state_df.resample('ME').mean().dropna()

    # 🔥 Create ML features
    state_df = create_features(state_df)

    # Train-test split
    split_date = '2025-01-01'
    train = state_df[state_df.index < split_date]
    test = state_df[state_df.index >= split_date]

    if len(test) == 0:
        print("⚠️ No test data for 2025, skipping...\n")
        continue

    try:
        # Features and target
        X_train = train.drop(columns=['Price'])
        y_train = train['Price']

        X_test = test.drop(columns=['Price'])
        y_test = test['Price']

        # 🔥 Random Forest model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluation
        mae = mean_absolute_error(y_test, predictions)
        print(f"✅ MAE: {mae:.2f}")

        # =========================
        # 📈 FULL GRAPH
        # =========================
        plt.figure(figsize=(10,5))

        plt.plot(state_df.index, state_df['Price'], label='Actual (Full)')
        plt.plot(test.index, predictions, label='Predicted')

        plt.axvline(pd.to_datetime(split_date), linestyle='--', label='Train/Test Split')

        plt.title(f"{state} Tomato Price Prediction (Random Forest)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"random_forest_model/{state}_rf_full.png")
        plt.close()

        # =========================
        # 🔍 ZOOM GRAPH
        # =========================
        plt.figure(figsize=(10,5))

        plt.plot(test.index, y_test, label='Actual')
        plt.plot(test.index, predictions, label='Predicted')

        plt.title(f"{state} Random Forest Prediction (Zoomed)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"random_forest_model/{state}_rf_zoom.png")
        plt.close()

        print(f"📊 Random Forest graphs saved for {state}\n")

    except Exception as e:
        print(f"❌ Error in {state}: {e}\n")

print("🎯 Random Forest Modeling Complete!")