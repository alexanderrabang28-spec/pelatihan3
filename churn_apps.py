
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load Machine Learning Model
model = joblib.load('alexander_churn_pipeline.joblib')


def main():
    st.title('❤️ Heart Disease Prediction')
    st.write('Heart Disease Prediction Model Deployment')

    st.subheader('Input Data Pasien')

    # Input features
    id = st.number_input("Patient ID", 0, 100000, step=1)

    age = st.number_input("Age (years)", 1, 120, step=1)

    sex = st.radio(
        "Sex",
        ["Male", "Female"]
    )

    dataset = st.selectbox(
        "Dataset",
        ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        ["typical angina", "atypical angina",
         "non-anginal", "asymptomatic"]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        0, 300, step=1
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        0, 1000, step=1
    )

    fbs = st.radio(
        "Fasting Blood Sugar > 120 mg/dl",
        ["True", "False"]
    )

    restecg = st.selectbox(
        "Resting ECG",
        ["normal", "st-t abnormality", "lv hypertrophy"]
    )

    thalch = st.number_input(
        "Maximum Heart Rate Achieved",
        0, 300, step=1
    )

    exang = st.radio(
        "Exercise Induced Angina",
        ["True", "False"]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        0.0, 20.0, step=0.1
    )

    # Prediction target is num
    data = {
        'id': int(id),
        'age': int(age),
        'sex': sex,
        'dataset': dataset,
        'cp': cp,
        'trestbps': int(trestbps),
        'chol': int(chol),
        'fbs': fbs == "True",
        'restecg': restecg,
        'thalch': int(thalch),
        'exang': exang == "True",
        'oldpeak': float(oldpeak)
    }

    # Create DataFrame
    df = pd.DataFrame([data])

    # Make Prediction
    if st.button("Make Prediction"):

        prediction = model.predict(df)[0]

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success("Prediction: No Heart Disease")
        else:
            st.warning(f"Prediction: Heart Disease (Class {prediction})")

        # Show input data
        with st.expander("View Patient Data"):
            st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
