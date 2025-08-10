# app.py - Streamlit app (prediction + SHAP explanation)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import io
import streamlit.components.v1 as components

st.set_page_config(page_title="Term Deposit Prediction", layout="wide")
st.title("📈 Bank Term Deposit Subscription Prediction")
st.write("Enter customer data below and click Predict. The app returns a prediction, probability and a SHAP explanation.")

@st.cache_resource
def load_model(path="rf_pipeline.joblib"):
    return joblib.load(path)

try:
    model = load_model("rf_pipeline.joblib")
except Exception as e:
    st.error(f"Could not load rf_pipeline.joblib. Make sure the file is in the app folder. Error: {e}")
    st.stop()

# Get preprocessor to extract feature names if needed
preprocessor = model.named_steps['pre']
clf = model.named_steps['clf']

# --- Build form inputs (main fields from dataset) ---
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("age", min_value=18, max_value=120, value=35)
        job = st.selectbox("job", ["admin.", "blue-collar", "technician", "services", "management",
                                   "retired", "self-employed", "unemployed", "student", "housemaid",
                                   "entrepreneur", "unknown"])
        marital = st.selectbox("marital", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("education", ["basic.4y","basic.6y","basic.9y","high.school",
                                               "illiterate","professional.course","university.degree","unknown"])
    with col2:
        default = st.selectbox("default", ["no","yes","unknown"])
        housing = st.selectbox("housing", ["no","yes","unknown"])
        loan = st.selectbox("loan", ["no","yes","unknown"])
        contact = st.selectbox("contact", ["cellular","telephone"])
    with col3:
        month = st.selectbox("month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
        day_of_week = st.selectbox("day_of_week", ["mon","tue","wed","thu","fri"])
        campaign = st.number_input("campaign (contacts during this campaign)", min_value=1, value=1)
        pdays = st.number_input("pdays (days since last contact, -1 if none)", value=-1)
    # Additional fields below form
    previous = st.number_input("previous (contacts before this campaign)", min_value=0, value=0)
    poutcome = st.selectbox("poutcome", ["nonexistent","failure","success"])
    emp_var_rate = st.number_input("emp.var.rate", value=1.0, format="%.2f")
    cons_price_idx = st.number_input("cons.price.idx", value=93.0, format="%.2f")
    cons_conf_idx = st.number_input("cons.conf.idx", value=-40.0, format="%.2f")
    euribor3m = st.number_input("euribor3m", value=4.0, format="%.3f")
    nr_employed = st.number_input("nr.employed", value=5000.0, format="%.1f")

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build input DataFrame exactly matching training features
    input_df = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education, "default": default,
        "housing": housing, "loan": loan, "contact": contact, "month": month, "day_of_week": day_of_week,
        "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
        "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx, "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m, "nr.employed": nr_employed
    }])

    # Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0,1]
    st.markdown(f"### Prediction: {'✅ Subscribed' if pred==1 else '❌ Not Subscribed'}")
    st.markdown(f"**Probability of subscription:** {proba:.2%}")

    # SHAP explanation
    st.write("### SHAP explanation (top contributors)")

    # Prepare background for explainer: try to sample from training data if available in pipeline metadata.
    try:
        # Transform input to preprocessed space
        X_pre = preprocessor.transform(input_df)  # shape (1, n_features)
        # Choose a small background: use the input itself (fallback) OR zeros if transform not possible
        background = X_pre  # using single-row background is acceptable though not ideal
    except Exception:
        # fallback to zeros matrix with appropriate width
        try:
            dummy = preprocessor.transform(pd.DataFrame([input_df.iloc[0]]))
            background = dummy
        except Exception:
            st.warning("Could not construct background for SHAP; explanation may be approximate.")
            background = np.zeros((1, preprocessor.transform(input_df).shape[1]))

    # Build Explainer (modern API) using classifier and background
    try:
        explainer = shap.Explainer(clf, background)
        shap_exp = explainer(X_pre)  # Explanation object
    except Exception as e:
        st.error(f"Failed to build SHAP explainer: {e}")
        shap_exp = None

    # Try interactive force plot (HTML)
    if shap_exp is not None:
        try:
            # shap.plots.force returns a matplotlib/HTML object based on display mode.
            # The following generates HTML (JS) output which we render inside Streamlit.
            force_html = shap.plots.force(shap_exp[0], matplotlib=False, show=False)
            # For shap v0.20+ the return value may be a JS HTML object with .data attribute
            html_data = None
            if hasattr(force_html, 'data'):
                html_data = force_html.data
            else:
                # try str(force_html)
                html_data = str(force_html)
            components.html(html_data, height=300, scrolling=True)
        except Exception:
            # Fallback: waterfall matplotlib -> convert to PNG and display
            try:
                fig, ax = plt.subplots(figsize=(8,3))
                shap.plots.waterfall(shap_exp[0], max_display=12, show=False)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                st.image(buf)
                plt.close(fig)
            except Exception as e:
                st.write("Could not render SHAP plots inline. Showing top contributors as text.")
                try:
                    vals = shap_exp.values[0]
                    # Map shap values to feature names
                    feat_names = preprocessor.get_feature_names_out()
                    s = pd.Series(vals, index=feat_names).sort_values(ascending=False)
                    st.write("Top positive contributors:")
                    st.write(s.head(10))
                    st.write("Top negative contributors:")
                    st.write(s.tail(10))
                except Exception as ex:
                    st.write("SHAP explanation is not available:", ex)
