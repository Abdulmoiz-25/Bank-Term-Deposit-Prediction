# 📘 DeveloperHub Task 6 – Bank Term Deposit Subscription Prediction

## 📌 Task Objective
Build a machine learning model and deploy a web app to predict whether a bank customer will subscribe to a term deposit, using customer data and economic indicators. The app also explains predictions via SHAP waterfall plots.

---

## 📁 Dataset
- **Name**: Bank Marketing Dataset  
- **Source**: UCI Machine Learning Repository (commonly used in term deposit prediction)  
- **Features** include:
  - Customer demographics (age, job, marital, education, etc.)  
  - Campaign details (contact method, month, day_of_week, campaign count)  
  - Economic indicators (emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed)  
  - Outcome variables and others  

---

## 🛠️ Tools & Libraries Used
- **Pandas** – data manipulation  
- **Numpy** – numerical operations  
- **Scikit-learn** – machine learning modeling (Random Forest pipeline)  
- **SHAP** – explainable AI for feature impact visualization  
- **Streamlit** – interactive app deployment  
- **PIL (Pillow)** – image handling for SHAP waterfall plots  

---

## 🚀 Approach

### 🔍 1. Data Preparation & Feature Engineering
- Cleaned and preprocessed raw customer data  
- Selected relevant features excluding call duration for prediction  

### 🤖 2. Model Training
- Built a Random Forest classifier pipeline with preprocessing  
- Trained and validated model on historical data  

### 🧪 3. Explanation & Visualization
- Generated SHAP waterfall plots for key test cases  
- Saved waterfall images and integrated into Streamlit app for interpretation  

### 🌐 4. Deployment
- Developed a Streamlit app with:
  - User input form for customer features  
  - Prediction and probability display  
  - SHAP waterfall plot image showing top contributors closest to the prediction  
  - Tables listing top positive and negative feature contributions  

---

## 📊 Results & Insights
- Model predicts term deposit subscription with reasonable accuracy  
- SHAP plots provide explainability highlighting influential features  
- Economic indicators and campaign variables are highly influential  

---

## ✅ Conclusion
The project demonstrates an end-to-end ML workflow: from data processing, modeling, explainability, to deployment on Streamlit Cloud, enabling interactive prediction and interpretation.

---

## 🌐 Live App
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bank-term-deposit-prediction-gdwrxx3amhngwuumykde9r.streamlit.app/)

---

## 📚 Useful Links
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)  
- [Streamlit Documentation](https://docs.streamlit.io/)  

---

> 🔖 Submitted as part of the **Developer Hub Internship Program**
)
