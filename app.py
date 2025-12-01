import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------
#                PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction Dashboard")
st.write("Analyze customer churn patterns and predict the likelihood of customer churn using Machine Learning.")

# -----------------------------------------------------
#                LOAD DATA
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Customer Churn.csv")
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0").astype("float")
    df["SeniorCitizen"] = df["SeniorCitizen"].apply(lambda x: "Yes" if x == 1 else "No")
    return df

df = load_data()

# -----------------------------------------------------
#                SIDEBAR NAVIGATION
# -----------------------------------------------------
menu = st.sidebar.radio(
    "Choose a Section",
    ["Dataset Overview", "EDA", "Churn Prediction"]
)

# -----------------------------------------------------
#               SECTION 1 — Dataset Overview
# -----------------------------------------------------
if menu == "Dataset Overview":
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(df.describe())

    st.subheader("🔢 Null Values")
    st.write(df.isnull().sum())

# -----------------------------------------------------
#               SECTION 2 — EDA
# -----------------------------------------------------
elif menu == "EDA":
    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    # Churn distribution
    with col1:
        st.subheader("Churn Count")
        fig, ax = plt.subplots()
        sns.countplot(x="Churn", data=df)
        st.pyplot(fig)

    with col2:
        st.subheader("Churn Percentage")
        churn_rate = df["Churn"].value_counts(normalize=True) * 100
        fig, ax = plt.subplots()
        plt.pie(churn_rate, labels=churn_rate.index, autopct="%1.2f%%")
        st.pyplot(fig)

    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x="Contract", hue="Churn", data=df)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Tenure Distribution by Churn")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=72)
    st.pyplot(fig)

# -----------------------------------------------------
#               SECTION 3 — CHURN PREDICTION
# -----------------------------------------------------
elif menu == "Churn Prediction":
    st.header("🤖 ML-Based Churn Prediction")

    # -------------------------------
    #     DATA PREPROCESSING
    # -------------------------------
    df_model = df.copy()

    # ❗ Remove customerID (not useful + causes mismatch)
    df_model = df_model.drop("customerID", axis=1)

    # Label Encoder dictionary
    label_encoders = {}

    # Encode categorical cols
    for col in df_model.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    # Separate features + target
    X = df_model.drop("Churn", axis=1)
    y = df_model["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save correct feature order after encoding
    feature_order = list(X_train.columns)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    st.success("Model trained successfully!")

    # -------------------------------
    #      SIDEBAR INPUTS
    # -------------------------------
    st.sidebar.subheader("Enter Customer Details")

    gender = st.sidebar.selectbox("Gender", df["gender"].unique())
    senior = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.sidebar.selectbox("Partner", df["Partner"].unique())
    dependents = st.sidebar.selectbox("Dependents", df["Dependents"].unique())
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 10)
    phone = st.sidebar.selectbox("Phone Service", df["PhoneService"].unique())
    internet = st.sidebar.selectbox("Internet Service", df["InternetService"].unique())
    contract = st.sidebar.selectbox("Contract", df["Contract"].unique())
    payment = st.sidebar.selectbox("Payment Method", df["PaymentMethod"].unique())
    monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.sidebar.number_input("Total Charges", 0.0, 9000.0, 1000.0)

    services = {}
    for col in [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "PaperlessBilling"
    ]:
        services[col] = st.sidebar.selectbox(col, df[col].unique())

    # -------------------------------
    #     CREATE INPUT DATAFRAME
    # -------------------------------
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    input_data.update(services)

    input_df = pd.DataFrame([input_data])

    # -------------------------------
    #     APPLY LABEL ENCODING
    # -------------------------------
    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # -------------------------------
    #    ENSURE SAME COLUMN ORDER
    # -------------------------------
    input_df = input_df[feature_order]

    # -------------------------------
    #       PREDICTION BUTTON
    # -------------------------------
    if st.button("Predict Churn"):
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("🔍 Prediction Result:")
        if pred == 1:
            st.error(f"❗ Customer is LIKELY TO CHURN (Risk Score: {prob:.2f})")
        else:
            st.success(f"✅ Customer is NOT likely to churn (Risk Score: {prob:.2f})")

        # Recommendations
        st.subheader("📌 Recommendation")
        if pred == 1:
            st.write("""
            ### Suggested Actions:
            - Offer discounts or loyalty perks  
            - Provide technical support  
            - Promote long-term contract benefits  
            - Improve service quality  
            """)
        else:
            st.write("Customer shows strong retention behavior.")

