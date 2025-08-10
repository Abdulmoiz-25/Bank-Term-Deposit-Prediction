import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import shap
import matplotlib.pyplot as plt
import io
import gzip
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="Term Deposit Prediction", layout="wide")
st.title("📈 Bank Term Deposit Subscription Prediction")
st.write("Enter customer data below and click Predict. The app returns a prediction, probability and a SHAP explanation.")

# =========================
# Load model with gzip + cloudpickle
# =========================
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
    st.error(f"Could not load rf_pipeline_cloud.pkl.gz. Make sure the file is in the app folder. Error: {e}")
    st.stop()

shap_bg = load_shap_bg()

# Extract preprocessor and classifier
preprocessor = model.named_steps['pre']
clf = model.named_steps['clf']

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("age", 18, 120, 35) # Changed value to 35
        job = st.selectbox("job", ["admin.", "blue-collar", "technician", "services", "management",
                              "retired", "self-employed", "unemployed", "student", "housemaid",
                              "entrepreneur", "unknown"], index=9) # Changed index to 9 (student)
        marital = st.selectbox("marital", ["married", "single", "divorced", "unknown"], index=1) # Changed index to 1 (single)
        education = st.selectbox("education", ["basic.4y", "basic.6y", "basic.9y", "high.school",
                                          "illiterate", "professional.course", "university.degree", "unknown"], index=6) # Kept index 6 (university.degree)
    with col2:
        default = st.selectbox("default", ["no", "yes", "unknown"], index=0) # Kept index 0 (no)
        housing = st.selectbox("housing", ["no", "yes", "unknown"], index=1) # Kept index 1 (yes)
        loan = st.selectbox("loan", ["no", "yes", "unknown"], index=0) # Kept index 0 (no)
        contact = st.selectbox("contact", ["cellular", "telephone"], index=0) # Kept index 0 (cellular)
    with col3:
        month = st.selectbox("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], index=2) # Changed index to 2 (mar)
        day_of_week = st.selectbox("day_of_week", ["mon", "tue", "wed", "thu", "fri"], index=3) # Kept index 3 (thu)
        campaign = st.number_input("campaign", 1, 50, 1) # Changed value to 1
        pdays = st.number_input("pdays", -1, 999, 5) # Changed value to 5
    previous = st.number_input("previous", 0, 50, 1) # Changed value to 1
    poutcome = st.selectbox("poutcome", ["nonexistent", "failure", "success"], index=2) # Kept index 2 (success)
    duration = st.number_input("duration", 0, 10000, 2000) # Changed value to 2000
    emp_var_rate = st.number_input("emp.var.rate", value=-1.8, format="%.2f") # Changed value to -1.8
    cons_price_idx = st.number_input("cons.price.idx", value=92.893, format="%.3f") # Changed value to 92.893
    cons_conf_idx = st.number_input("cons.conf.idx", value=-20.0, format="%.1f") # Changed value to -20.0
    euribor3m = st.number_input("euribor3m", value=0.7, format="%.3f") # Changed value to 0.7
    nr_employed = st.number_input("nr.employed", value=5099.1, format="%.1f") # Changed value to 5099.1
    balance = st.number_input("balance", value=1000) # Changed value to 1000
    submitted = st.form_submit_button("Predict")

if submitted:
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
        "duration": duration,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        # Added 'balance' with a default value to match the expected 21 features
        # If your model expects a different 21st feature, replace 'balance' with its name.
        "balance": balance
    }])
    # Predict using the full pipeline (which applies preprocessing)
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0, 1]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    st.markdown(f"### Prediction: {'✅ Subscribed' if pred == 1 else '❌ Not Subscribed'}")
    st.markdown(f"**Probability of subscription:** {proba:.2%}")

    # Prepare SHAP explanation
    try:
        X_pre = preprocessor.transform(input_df)
    except Exception as e:
        st.error(f"Error preprocessing input for SHAP: {e}")
        X_pre = None

    if X_pre is not None:
        background = shap_bg if shap_bg is not None else X_pre
        try:
            explainer = shap.Explainer(clf, background)
            shap_exp = explainer(X_pre)
            try:
                force_html = shap.plots.force(shap_exp[0], matplotlib=False, show=False)
                html_data = getattr(force_html, 'data', str(force_html))
                components.html(html_data, height=300, scrolling=True)
            except Exception:
                fig, ax = plt.subplots(figsize=(8, 3))
                shap.plots.waterfall(shap_exp[0], max_display=12, show=False)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                st.image(buf)
                plt.close(fig)
        except Exception as e:
            st.write("Could not render SHAP plots inline. Showing top contributors as text.")
            try:
                vals = shap_exp.values[0] if shap_exp.values.ndim != 3 else shap_exp.values[0][:, 1]
                feat_names = preprocessor.get_feature_names_out()
                s = pd.Series(vals, index=feat_names).sort_values(ascending=False)
                st.write("Top positive contributors:")
                st.write(s.head(10))
                st.write("Top negative contributors:")
                st.write(s.tail(10))
            except Exception as ex:
                st.write("SHAP explanation is not available:", ex)
