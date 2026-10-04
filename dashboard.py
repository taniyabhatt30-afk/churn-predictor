import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Predictor")

# Load model and data once
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")
df = pd.read_csv("churn_features.csv")

X = df[model_columns]
y = df['Churn']
df['Churn_Risk'] = model.predict_proba(X)[:, 1]

# --- Three tabs ---
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "🔮 Predict Churn", "📈 Model Performance"])

# ============ TAB 1: DASHBOARD ============
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{y.mean()*100:.1f}%")
    col3.metric("High Risk Customers (>70%)", (df['Churn_Risk'] > 0.7).sum())

    st.divider()
    st.subheader("🔴 Top 10 customers at highest churn risk")
    top_risk = df.sort_values('Churn_Risk', ascending=False).head(10)
    st.dataframe(top_risk[['tenure', 'MonthlyCharges', 'Churn_Risk']])

    st.divider()
    st.subheader("📈 What drives churn overall")
    st.image("shap_summary.png")

# ============ TAB 2: PREDICT CHURN ============
with tab2:
    st.subheader("Enter customer details")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)
        total_charges = st.number_input("Total Charges ($)", 0, 10000, 840)
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    if st.button("Predict Churn"):
        # Build a single-row input matching the model's expected columns
        input_dict = dict.fromkeys(model_columns, 0)
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = total_charges

        if contract == "One year":
            input_dict['Contract_One year'] = 1
        elif contract == "Two year":
            input_dict['Contract_Two year'] = 1

        if internet == "Fiber optic":
            input_dict['InternetService_Fiber optic'] = 1
        elif internet == "No":
            input_dict['InternetService_No'] = 1

        if paperless == "Yes":
            input_dict['PaperlessBilling_Yes'] = 1

        input_df = pd.DataFrame([input_dict])[model_columns]

        prob = model.predict_proba(input_df)[0][1]
        prediction = "Likely to Churn" if prob > 0.5 else "Likely to Stay"

        if prob > 0.7:
            risk = "🔴 High"
        elif prob > 0.4:
            risk = "🟡 Medium"
        else:
            risk = "🟢 Low"

        st.divider()
        st.subheader("Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", prediction)
        col2.metric("Churn Probability", f"{prob*100:.1f}%")
        col3.metric("Risk Level", risk)

        st.info(f"This customer is on a **{contract}** contract with **{tenure} months** tenure and pays **${monthly_charges}/month**. Shorter tenure, month-to-month contracts, and higher charges are the biggest factors pushing risk up.")

# ============ TAB 3: MODEL PERFORMANCE ============
with tab3:
    st.subheader("Model Evaluation Metrics")
    st.write("Model: Logistic Regression (with StandardScaler, class_weight='balanced')")
    st.write("ROC-AUC: 0.835")
    st.write("Recall (churners): 0.80  |  Precision (churners): 0.49")

    st.divider()
    st.subheader("Feature Importance")
    st.image("shap_summary.png")