import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import catboost
# Load the model
model = joblib.load('catboost_model.pkl')




# Define feature names
feature_names = [
'Gestational_age','Maternal_age','Gravidity','Parturition','Prior_C-sections_number',
'Uterine_surgery_number','Cervical_canal_length','Uterine_anteroposterior_diameter_ratio', 
'Placental_abnormal_vasculature_diameter','Placental_abnormal_vasculature_area',
'Intraplacental_dark_T2_band_area'
]
# features = np.array([[40,274,7,1,1,0,4.3,1.21,2.3,2.783,4.0746]])
# print(model.predict_proba(input)[0][1])

# Streamlit user interface
st.title("Adverse Clinical Outcome")

# age: numerical input
Gestational_age = st.number_input("Gestational_age:", min_value=0, max_value=120, value=50)

# sex: categorical selection
Maternal_age = st.number_input("Maternal_age:",  min_value=0, max_value=120, value=50)

# cp: categorical selection
Gravidity = st.number_input("Gravidity:", min_value=0, max_value=120, value=50)

# trestbps: numerical input
Parturition = st.number_input("Parturition:", min_value=0, max_value=120, value=50)

# trestbps: numerical input
Prior_C_sections_number = st.number_input("Prior_C-sections_number:", min_value=0, max_value=120, value=50)

# chol: numerical input
Uterine_surgery_number = st.number_input("Uterine_surgery_number:", min_value=0, max_value=120, value=50)

# fbs: categorical selection
Cervical_canal_length = st.number_input("Cervical_canal_length:", min_value=0, max_value=120, value=50)

# restecg: categorical selection
Uterine_anteroposterior_diameter_ratio = st.number_input("Uterine_anteroposterior_diameter_ratio:", min_value=0, max_value=120, value=50)

# thalach: numerical input
Placental_abnormal_vasculature_diameter = st.number_input("Placental_abnormal_vasculature_diameter:", min_value=0, max_value=120, value=50)

# thalach: numerical input
Placental_abnormal_vasculature_area = st.number_input("Placental_abnormal_vasculature_area:",  min_value=0, max_value=120, value=50)

# exang: categorical selection
Intraplacental_dark_T2_band_area = st.number_input("Intraplacental_dark_T2_band_area:", min_value=0, max_value=120, value=50)

# Process inputs and make predictions
feature_values = [Gestational_age,Maternal_age,Gravidity,Parturition,Prior_C_sections_number,Uterine_surgery_number,Cervical_canal_length,Uterine_anteroposterior_diameter_ratio,Placental_abnormal_vasculature_diameter,Placental_abnormal_vasculature_area,Intraplacental_dark_T2_band_area]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_proba = model.predict_proba(features)[0][1]

    # Display prediction results
    st.write(f"**Prediction Probabilities:** {predicted_proba}")


    if predicted_proba >= 0.5:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {predicted_proba:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {predicted_proba:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(features, columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame(features, columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")