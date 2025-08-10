import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import shap
import matplotlib.pyplot as plt
import io
import gzip
import os

st.set_page_config(page_title="Term Deposit Prediction", layout="wide")
st.title("📈 Bank Term Deposit Subscription Prediction")
st.write("Enter customer data below and click Predict. The app returns a prediction, probability and a SHAP explanation.")

@st.cache_resource
def load_model(path="rf_pipeline_cloud.pkl.gz"):
    with gzip.open(path, "rb") as f:
        return cloudpickle.load(f)

@st.cache_resource
def load_shap_bg(path="shap_bg.npy"):
    if os.path.exists(path):
        return np.load(path)
    else:
        return None

try:
    model = load_model("rf_pipeline_cloud.pkl.gz")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

shap_bg = load_shap_bg()

preprocessor = model.named_steps['pre']
clf = model.named_steps['clf']

with st.form("input_form"):
    age = st.number_input("age", 18, 120, 40)
    job = st.selectbox("job", ["admin.", "blue-collar", "technician", "services", "management",
                              "retired", "self-employed", "unemployed", "student", "housemaid",
                              "entrepreneur", "unknown"], index=4)
    marital = st.selectbox("marital", ["married", "single", "divorced", "unknown"], index=0)
    education = st.selectbox("education", ["basic.4y", "basic.6y", "basic.9y", "high.school",
                                          "illiterate", "professional.course", "university.degree", "unknown"], index=6)
    default = st.selectbox("default", ["no", "yes", "unknown"], index=0)
    housing = st.selectbox("housing", ["no", "yes", "unknown"], index=1)
    loan = st.selectbox("loan", ["no", "yes", "unknown"], index=0)
    contact = st.selectbox("contact", ["cellular", "telephone"], index=0)
    month = st.selectbox("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], index=4)
    day_of_week = st.selectbox("day_of_week", ["mon", "tue", "wed", "thu", "fri"], index=3)
    campaign = st.number_input("campaign", 1, 50, 3)
    pdays = st.number_input("pdays", -1, 999, 100)
    previous = st.number_input("previous", 0, 50, 5)
    poutcome = st.selectbox("poutcome", ["nonexistent", "failure", "success"], index=2)
    duration = st.number_input("duration", 0, 10000, 1000)
    emp_var_rate = st.number_input("emp.var.rate", value=1.4, format="%.2f")
    cons_price_idx = st.number_input("cons.price.idx", value=93.994, format="%.3f")
    cons_conf_idx = st.number_input("cons.conf.idx", value=-36.4, format="%.1f")
    euribor3m = st.number_input("euribor3m", value=4.857, format="%.3f")
    nr_employed = st.number_input("nr.employed", value=5191.0, format="%.1f")
    balance = st.number_input("balance", value=0)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
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
        "duration": duration,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "balance": balance
    }

    input_df = pd.DataFrame([input_dict])

    st.write("### Input Data:")
    st.write(input_df)

    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0,1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # DEBUG: force subscribed output to test waterfall plot
    # pred = 1
    # proba = 0.85

    st.markdown(f"### Prediction: {'✅ Subscribed' if pred == 1 else '❌ Not Subscribed'}")
    st.markdown(f"**Probability of subscription:** {proba:.2%}")

    try:
        X_pre = preprocessor.transform(input_df)
        background = shap_bg if shap_bg is not None else X_pre
        explainer = shap.Explainer(clf, background)
        shap_exp = explainer(X_pre)

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_exp[0], max_display=12, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        st.image(buf)
        plt.close(fig)
    except Exception as e:
        st.error(f"SHAP plotting error: {e}")
