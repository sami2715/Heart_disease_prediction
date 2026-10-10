
import pickle
import streamlit as st
import numpy as np

# Load the saved model and scaler
with open("svc_trained_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Configure the page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# App heading
st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient features used by your trained model."
)

st.info(
    "Educational demonstration only. This app cannot diagnose "
    "heart disease or replace professional medical assessment."
)

# Patient input form
with st.form("patient_form"):
    st.subheader("Patient Information")

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        index=2
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=0.0,
        max_value=700.0,
        value=233.0
    )

    thalach = st.number_input(
        "Maximum Heart Rate (thalach)",
        min_value=0.0,
        max_value=300.0,
        value=150.0
    )

    thal = st.selectbox(
        "Thalassemia Type (thal)",
        options=[0, 1, 2, 3],
        index=2
    )

    submitted = st.form_submit_button("Predict")

# Run prediction
if submitted:
    try:
        patient_data = np.array(
            [[cp, chol, thalach, thal]],
            dtype=float
        )

        scaled_data = scaler.transform(patient_data)
        prediction = model.predict(scaled_data)[0]

        st.subheader("Prediction Result")

        # Assumes 0 = no disease and 1 = disease.
        # Verify these labels against your training notebook.
        if prediction == 0:
            st.success("✅ No Heart Disease Detected")
        elif prediction == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.write(f"Model output: {prediction}")

        st.caption(
            "This is a machine-learning prediction, not a "
            "medical diagnosis. Consult a qualified healthcare "
            "professional about medical concerns."
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")
        st.warning(
            "Check the model's feature count, feature order, "
            "and preprocessing against your training notebook."
        )
