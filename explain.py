import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("churn_features.csv")
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Rebuild and train the same pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=5000, class_weight='balanced'))
])
model.fit(X_train, y_train)

# --- SHAP setup ---
# SHAP sees the same numbers the logistic regression actually used
X_test_scaled = model.named_steps['scaler'].transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# LinearExplainer is the correct SHAP explainer for Logistic Regression
explainer = shap.LinearExplainer(model.named_steps['logistic'], X_test_scaled)
shap_values = explainer.shap_values(X_test_scaled)

# Overall feature importance chart - which features matter most across ALL customers
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')
plt.close()

print("Saved shap_summary.png")

# Save the trained model + column order, so the dashboard can reuse it
joblib.dump(model, "churn_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Saved churn_model.pkl and model_columns.pkl")