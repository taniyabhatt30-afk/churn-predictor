import pandas as pd

# Load the cleaned data (output of clean.py)
df = pd.read_csv("churn_clean.csv")

# One-hot encode all text/categorical columns automatically
# drop_first=True avoids redundant columns (prevents multicollinearity)
df_encoded = pd.get_dummies(df, drop_first=True)

# Show the new column names so you can see what got created
print("Total columns after encoding:", len(df_encoded.columns))
print(df_encoded.columns.tolist())

# Save this as the final feature-ready dataset
df_encoded.to_csv("churn_features.csv", index=False)

print("Feature engineering done. Saved as churn_features.csv")
print("New shape:", df_encoded.shape)