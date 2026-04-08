import pandas as pd
import os

base_folder = "Tomato_Data"
all_data = []

for state_folder in os.listdir(base_folder):
    state_path = os.path.join(base_folder, state_folder)
    
    if os.path.isdir(state_path):
        state_name = state_folder.split('_')[0]
        
        print(f"\nProcessing State: {state_name}")
        
        for file in os.listdir(state_path):
            if file.endswith(".csv"):
                file_path = os.path.join(state_path, file)
                print(f"  → File: {file}")
                
                try:
                    # 🔥 Skip first row (garbage title)
                    df = pd.read_csv(file_path, skiprows=1)

                    # Clean column names
                    df.columns = [col.strip() for col in df.columns]

                    # Select correct columns
                    df = df[['Price Date', 'Modal Price']]

                    # Rename
                    df.columns = ['Date', 'Price']

                    # 🔥 Remove commas from price
                    df['Price'] = df['Price'].astype(str).str.replace(",", "")

                    # Convert types
                    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
                    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

                    # Drop bad rows
                    df = df.dropna()

                    # Add state
                    df['State'] = state_name

                    all_data.append(df)

                except Exception as e:
                    print(f"Error in {file}: {e}")

# Safety check
if not all_data:
    print("❌ No data found!")
    exit()

# Combine
final_df = pd.concat(all_data)

# Sort
final_df = final_df.sort_values(by='Date')

# Aggregate (IMPORTANT)
final_df = final_df.groupby(['Date', 'State'])['Price'].mean().reset_index()

# Save
final_df.to_csv("cleaned_tomato_prices.csv", index=False)

print("\n✅ DONE BROOO 🔥")
print(final_df.head())



import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

for state in final_df['State'].unique():
    temp = final_df[final_df['State'] == state]
    plt.plot(temp['Date'], temp['Price'], label=state)

plt.legend()
plt.title("Tomato Prices by State")
plt.show()