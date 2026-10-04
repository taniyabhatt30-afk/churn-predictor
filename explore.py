import pandas as pd

df = pd.read_csv("churn_data.csv")

print(df.info())          # shows column names, types, missing values
print(df.describe())      # stats for numeric columns
print(df['Churn'].value_counts())   # how many churned vs not

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='tenure', hue='Churn', data=df)
plt.title("Churn by Tenure")
plt.show()