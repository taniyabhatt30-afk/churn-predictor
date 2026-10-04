import pandas as pd

# Load the original data (not the cleaned one - we're creating that now)
df = pd.read_csv("churn_data.csv")

# Fix TotalCharges: it's stored as text, convert to numbers
# Anything that fails to convert becomes NaN (missing)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many rows became missing after conversion
print("Missing values after conversion:", df['TotalCharges'].isnull().sum())

# Drop those few broken rows
df = df.dropna(subset=['TotalCharges'])

# Drop customerID - it's just an ID, not useful for predicting churn
df = df.drop('customerID', axis=1)

# Convert target column from Yes/No text to 1/0 numbers
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Save the cleaned result as a new file
df.to_csv("churn_clean.csv", index=False)

print("Cleaning done. Saved as churn_clean.csv")
print("New shape:", df.shape)