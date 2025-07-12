# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from xgboost_model import run_xgboost_model
from code2 import run_classical_models

# Streamlit configuration
st.set_page_config(page_title="Antibiotic Resistance Dashboard", layout="wide")
st.title("🧬 Antibiotic Resistance Prediction Dashboard")

# Sidebar: Antibiotic selection
antibiotic = st.sidebar.selectbox(
    "Select Antibiotic", 
    ['azm_sr', 'cip_sr', 'cfx_sr'], 
    format_func=lambda x: x.upper().replace('_SR', '')
)

# Run models
st.info(f"Running models for: **{antibiotic.upper().replace('_SR', '')}**")

with st.spinner("⏳ Loading models..."):
    xgb_result = run_xgboost_model(antibiotic)
    classical_results = run_classical_models(antibiotic)

all_results = classical_results + [xgb_result]

# Accuracy comparison
st.header("📊 Accuracy Comparison")

acc_df = pd.DataFrame([{
    'Model': res['model'],
    'Train Accuracy': res['train_accuracy'],
    'Test Accuracy': res['test_accuracy']
} for res in all_results])

st.dataframe(acc_df.set_index("Model"))

# Bar plot
fig, ax = plt.subplots()
acc_df.set_index("Model")[["Train Accuracy", "Test Accuracy"]].plot.bar(ax=ax, figsize=(8, 5))
plt.ylim(0.5, 1.05)
plt.title("Train vs Test Balanced Accuracy")
plt.ylabel("Balanced Accuracy")
plt.xticks(rotation=0)
st.pyplot(fig)

# Confusion matrices
st.header("🧮 Confusion Matrices")
cols = st.columns(len(all_results))
for i, res in enumerate(all_results):
    with cols[i]:
        st.subheader(res['model'])
        cm = res.get('confusion_matrix')
        if cm is not None:
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Sensitive', 'Resistant'],
                        yticklabels=['Sensitive', 'Resistant'],
                        ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            st.pyplot(fig_cm)
        else:
            st.warning("Confusion matrix not available.")

# Feature importance
st.header("🌟 Top 5 Features by Model")
for res in all_results:
    fi = res.get('feature_importance')
    if fi is not None:
        st.subheader(res['model'])
        st.dataframe(fi.head(5).reset_index(drop=True))
    else:
        st.warning(f"No feature importance available for {res['model']}.")

# Optional: CV score and best params (XGBoost only)
if 'cv_score' in xgb_result:
    st.subheader("🔧 XGBoost Training Info")
    st.write(f"**Best CV Balanced Accuracy**: `{xgb_result['cv_score']:.4f}`")
    st.write("**Best Hyperparameters:**")
    st.json(xgb_result['best_params'])

# Footer
st.markdown("---")
st.markdown("Made by **Karthik R S** | Powered by Scikit-learn, XGBoost, Streamlit")
