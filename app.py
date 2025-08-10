import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import gzip
import os
from PIL import Image

st.set_page_config(page_title="Term Deposit Prediction", layout="wide")
st.title("📈 Bank Term Deposit Subscription Prediction")
st.write("Enter customer data below and click Predict. The app returns prediction, probability, and SHAP explanation images with top contributors.")

# Load model and shap background
@st.cache_resource
def load_model(path="rf_pipeline_cloud.pkl.gz"):
    with gzip.open(path, "rb") as f:
        return cloudpickle.load(f)

@st.cache_resource
def load_shap_bg(path="shap_bg.npy"):
    if os.path.exists(path):
        return np.load(path)
    return None

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

shap_bg = load_shap_bg()

# Extract preprocessor and classifier
preprocessor = model.named_steps['pre']
clf = model.named_steps['clf']

# Your test cases with SHAP values and matching waterfall plot images in images folder
test_cases = {
    6373: "images/waterfall_6373.png",
    3615: "images/waterfall_3615.png",
    5391: "images/waterfall_5391.png",
    734:  "images/waterfall_734.png",
    3567: "images/waterfall_3567.png"
}

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("age", 18, 120, 45)
        job = st.selectbox("job", ["admin.", "blue-collar", "technician", "services", "management",
                                  "retired", "self-employed", "unemployed", "student", "housemaid",
                                  "entrepreneur", "unknown"], index=5)
        marital = st.selectbox("marital", ["married", "single", "divorced", "unknown"], index=1)
        education = st.selectbox("education", ["basic.4y", "basic.6y", "basic.9y", "high.school",
                                              "illiterate", "professional.course", "university.degree", "unknown"], index=6)
    with col2:
        default = st.selectbox("default", ["no", "yes", "unknown"], index=0)
        housing = st.selectbox("housing", ["no", "yes", "unknown"], index=0)
        loan = st.selectbox("loan", ["no", "yes", "unknown"], index=0)
        contact = st.selectbox("contact", ["cellular", "telephone"], index=0)
    with col3:
        month = st.selectbox("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                                      "sep", "oct", "nov", "dec"], index=9)
        day_of_week = st.selectbox("day_of_week", ["mon", "tue", "wed", "thu", "fri"], index=3)
        campaign = st.number_input("campaign", 1, 50, 1)
        pdays = st.number_input("pdays", -1, 999, 3)

    previous = st.number_input("previous", 0, 50, 1)
    poutcome = st.selectbox("poutcome", ["nonexistent", "failure", "success"], index=2)
    emp_var_rate = st.number_input("emp.var.rate", value=-2.9, format="%.2f")
    cons_price_idx = st.number_input("cons.price.idx", value=92.201, format="%.3f")
    cons_conf_idx = st.number_input("cons.conf.idx", value=-31.4, format="%.1f")
    euribor3m = st.number_input("euribor3m", value=0.8, format="%.3f")
    nr_employed = st.number_input("nr.employed", value=5000.0, format="%.1f")
    balance = st.number_input("balance", value=5000)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build input DataFrame WITHOUT duration
    input_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "balance": balance
    }])

    # Show input for user confirmation
    st.write("### Input data for prediction:")
    st.dataframe(input_df)

    # Predict
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0, 1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown(f"### Prediction: {'✅ Subscribed' if pred == 1 else '❌ Not Subscribed'}")
    st.markdown(f"**Probability of subscription:** {proba:.2%}")

    # Map of example test case probabilities
    test_cases_probs = {
        6373: 0.0,
        3615: 0.655,
        5391: 0.015,
        734: 0.765,
        3567: 0.020
    }

    # Find closest test case by probability
    closest_case = min(test_cases_probs.keys(), key=lambda k: abs(test_cases_probs[k] - proba))
    img_path = test_cases[closest_case]

    st.write(f"Showing SHAP waterfall plot closest to your prediction (Test case index: {closest_case})")

    try:
        img = Image.open(img_path)
        st.image(img, caption=f"Waterfall plot for test case {closest_case}")
    except FileNotFoundError:
        st.warning(f"Waterfall plot image '{img_path}' not found. Please upload it in the app folder.")

    # Top positive SHAP contributors
    shap_top_pos = {
        6373: {
            "month_may": 0.005538,
            "day_of_week_thu": 0.003557,
            "default_unknown": 0.002507,
            "age": 0.002357,
            "default_no": 0.001822,
            "contact_telephone": 0.001742,
            "campaign": 0.001638,
            "job_technician": 0.001536
        },
        3615: {
            "euribor3m": 0.124435,
            "nr.employed": 0.117794,
            "emp.var.rate": 0.080181,
            "cons.conf.idx": 0.051238,
            "cons.price.idx": 0.033215,
            "month_apr": 0.024327,
            "age": 0.021268,
            "month_may": 0.012516
        },
        5391: {
            "contact_telephone": 0.007689,
            "contact_cellular": 0.005245,
            "cons.price.idx": 0.004468,
            "education_university.degree": 0.001965,
            "month_aug": 0.001800,
            "default_no": 0.001708,
            "job_technician": 0.001443,
            "day_of_week_mon": 0.001342
        },
        734: {
            "nr.employed": 0.140340,
            "euribor3m": 0.135097,
            "emp.var.rate": 0.084644,
            "contact_telephone": 0.042504,
            "contact_cellular": 0.039225,
            "cons.price.idx": 0.031847,
            "day_of_week_wed": 0.024666,
            "month_sep": 0.019837
        },
        3567: {
            "contact_telephone": 0.006171,
            "contact_cellular": 0.005295,
            "job_management": 0.004231,
            "month_may": 0.003308,
            "cons.price.idx": 0.002849,
            "month_aug": 0.001977,
            "campaign": 0.001114,
            "cons.conf.idx": 0.000892
        }
    }

    # Top negative SHAP contributors
    shap_top_neg = {
        6373: {
            "pdays": -0.002824,
            "month_jul": -0.003148,
            "housing_yes": -0.005652,
            "cons.conf.idx": -0.014194,
            "euribor3m": -0.017322,
            "nr.employed": -0.020267,
            "month_aug": -0.023142,
            "emp.var.rate": -0.029946
        },
        3615: {
            "loan_no": -0.000862,
            "housing_yes": -0.000893,
            "month_jul": -0.000922,
            "poutcome_success": -0.001995,
            "education_university.degree": -0.002049,
            "day_of_week_wed": -0.002782,
            "pdays": -0.003282,
            "campaign": -0.007399
        },
        5391: {
            "housing_yes": -0.004516,
            "marital_divorced": -0.004614,
            "month_jul": -0.004626,
            "housing_no": -0.004856,
            "job_admin.": -0.005295,
            "euribor3m": -0.019849,
            "nr.employed": -0.023010,
            "emp.var.rate": -0.031801
        },
        734: {
            "month_jul": -0.000360,
            "month_mar": -0.000467,
            "job_retired": -0.000671,
            "month_dec": -0.000726,
            "day_of_week_tue": -0.000777,
            "poutcome_success": -0.001831,
            "pdays": -0.003253,
            "education_high.school": -0.015758
        },
        3567: {
            "default_unknown": -0.005656,
            "housing_yes": -0.006275,
            "default_no": -0.006315,
            "day_of_week_thu": -0.006734,
            "housing_no": -0.006773,
            "nr.employed": -0.017929,
            "emp.var.rate": -0.021213,
            "euribor3m": -0.022077
        }
    }

    st.write("### Top positive contributors:")
    st.table(pd.DataFrame(shap_top_pos[closest_case].items(), columns=["Feature", "SHAP Value"]).sort_values(by="SHAP Value", ascending=False))

    st.write("### Top negative contributors:")
    st.table(pd.DataFrame(shap_top_neg[closest_case].items(), columns=["Feature", "SHAP Value"]).sort_values(by="SHAP Value"))
