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

# Load model & shap bg
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

model = load_model()
shap_bg = load_shap_bg()

preprocessor = model.named_steps['pre']
clf = model.named_steps['clf']

expected_cols = preprocessor.feature_names_in_ if hasattr(preprocessor, 'feature_names_in_') else None
st.write("Expected features:", expected_cols)

# Manually define categorical features and their options (replace with your actual categories)
categorical_options = {
    "job": ["admin.", "blue-collar", "technician", "services", "management",
            "retired", "self-employed", "unemployed", "student", "housemaid",
            "entrepreneur", "unknown"],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": ["basic.4y", "basic.6y", "basic.9y", "high.school",
                  "illiterate", "professional.course", "university.degree", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["nonexistent", "failure", "success"]
}

# For all expected features, generate input widgets dynamically
inputs = {}

with st.form("input_form"):
    for feature in expected_cols:
        if feature in categorical_options:
            options = categorical_options[feature]
            inputs[feature] = st.selectbox(feature, options, index=0)
        else:
            # Numeric input fallback with some generic ranges/defaults
            # You can customize these for your features
            if feature in ["age"]:
                inputs[feature] = st.number_input(feature, min_value=18, max_value=120, value=40)
            elif feature in ["campaign", "previous"]:
                inputs[feature] = st.number_input(feature, min_value=0, max_value=50, value=1)
            elif feature in ["pdays"]:
                inputs[feature] = st.number_input(feature, min_value=-1, max_value=999, value=100)
            elif feature in ["duration"]:
                inputs[feature] = st.number_input(feature, min_value=0, max_value=10000, value=1000)
            elif feature in ["emp.var.rate"]:
                inputs[feature] = st.number_input(feature, value=1.4, format="%.2f")
            elif feature in ["cons.price.idx"]:
                inputs[feature] = st.number_input(feature, value=93.994, format="%.3f")
            elif feature in ["cons.conf.idx"]:
                inputs[feature] = st.number_input(feature, value=-36.4, format="%.1f")
            elif feature in ["euribor3m"]:
                inputs[feature] = st.number_input(feature, value=4.857, format="%.3f")
            elif feature in ["nr.employed"]:
                inputs[feature] = st.number_input(feature, value=5191.0, format="%.1f")
            else:
                # generic float input fallback
                inputs[feature] = st.number_input(feature, value=0.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([inputs])

    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0, 1]
        st.markdown(f"### Prediction: {'✅ Subscribed' if pred == 1 else '❌ Not Subscribed'}")
        st.markdown(f"**Probability of subscription:** {proba:.2%}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # SHAP explanation
    try:
        X_pre = preprocessor.transform(input_df)
        background = shap_bg if shap_bg is not None else X_pre
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
        st.write("SHAP explanation failed:", e)
