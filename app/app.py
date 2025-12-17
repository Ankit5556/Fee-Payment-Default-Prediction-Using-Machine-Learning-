# === app.py ===
import streamlit as st
import joblib, pickle
import pandas as pd

# load artifacts
rf1 = joblib.load("C:/Python Data Science/Internship Project(Fee Payment Default Prediction)/models/best_model_delayed_payment.pkl")
rf2 = joblib.load("C:/Python Data Science/Internship Project(Fee Payment Default Prediction)/models/best_model_needs_reminder.pkl")
scaler = joblib.load("C:/Python Data Science/Internship Project(Fee Payment Default Prediction)/models/scaler.pkl")
feature_cols = joblib.load("C:/Python Data Science/Internship Project(Fee Payment Default Prediction)/models/feature_columns.pkl")
label_encoders = pickle.load(open("C:/Python Data Science/Internship Project(Fee Payment Default Prediction)/models/label_encoders.pkl","rb"))

st.title("Fee Payment Default Prediction")

def options_for(col):
    return list(label_encoders[col].classes_)

# === USER INPUTS ===
admission_label = st.selectbox("Admission Status", options_for('admission_status'))
course_label    = st.selectbox("Course", options_for('course'))

# === RULE: If NOT admitted, disable fee inputs ===
if admission_label.lower() != "admitted":

    st.warning("❗ This student is NOT admitted — fee payment is not applicable.")

    total_fee = 0
    paid_fee = 0
    balance = 0

    #st.info(f"Total Fee: {total_fee} | Paid Fee: {paid_fee} | Balance: {balance}")

    # Unique button key here ↓
    if st.button("Predict", key="predict_not_admitted"):
        st.success("Delayed Payment: No")
        st.info("Needs Reminder: No")
        st.stop()

else:
    # === Admitted: enable inputs ===
    total_fee = st.number_input("Total Fee", min_value=0.0, value=10000.0, step=100.0)
    paid_fee  = st.number_input("Paid Fee", min_value=0.0, value=0.0, step=100.0)

    balance = total_fee - paid_fee
    st.info(f"Auto-calculated balance = {balance}")

# === NORMAL PREDICTION ===

# Unique button key here ↓
if st.button("Predict", key="predict_admitted"):

    df_row = pd.DataFrame([{
        'admission_status': admission_label,
        'total_fee': total_fee,
        'paid_fee': paid_fee,
        'balance': balance,
        'course': course_label
    }])

    # Encode categorical columns
    for col, le in label_encoders.items():
        if col in df_row.columns:
            df_row[col] = le.transform(df_row[col].astype(str))

    df_row = df_row[feature_cols]

    X_scaled = scaler.transform(df_row)

    p1 = rf1.predict(X_scaled)[0]
    p2 = rf2.predict(X_scaled)[0]

    delayed_prediction = "Yes" if int(p1)==1 else "No"
    reminder_prediction = "Yes" if int(p2)==1 else "No"

    # BUSINESS RULE: 70% Paid or less → Send reminder
    if total_fee > 0:
        paid_ratio = paid_fee / total_fee
        if paid_ratio <= 0.70:
            reminder_prediction = "Yes"
        else:
            reminder_prediction = "No"

    st.success(f"Delayed Payment: {delayed_prediction}")
    st.info(f"Needs Reminder: {reminder_prediction}")
    st.balloons()