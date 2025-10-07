import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zomato MMM Optimizer", page_icon="📊", layout="wide")
st.title("📊 Zomato Marketing Spend Optimizer")
st.markdown("Optimize your weekly marketing budget to drive maximum Zomato orders.")

@st.cache_resource
def load_model():
    return joblib.load("marketing_model.pkl")  # Make sure this matches your trained file

model = load_model()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Budget Allocation")

    total_budget = st.number_input("Total Weekly Budget (₹000)", min_value=0, value=200, step=10)

    facebook = st.slider("Facebook Spend", 0, 150, 50)
    google = st.slider("Google Spend", 0, 150, 50)
    influencer = st.slider("Influencer Spend", 0, 100, 30)
    tv = st.slider("TV Spend", 0, 200, 40)
    promo = st.slider("Promo Discount (%)", 0, 50, 15)

    current_total = facebook + google + influencer + tv
    utilization = (current_total / total_budget) * 100 if total_budget > 0 else 0

    if utilization > 100:
        st.error(f"🚨 Over budget by ₹{current_total - total_budget}k!")
    else:
        st.success(f"✅ Budget utilization: {utilization:.1f}%")

    st.markdown("#### Channel Share")
    if current_total > 0:
        st.write(f"Facebook: {(facebook/current_total)*100:.1f}%")
        st.write(f"Google: {(google/current_total)*100:.1f}%")
        st.write(f"Influencer: {(influencer/current_total)*100:.1f}%")
        st.write(f"TV: {(tv/current_total)*100:.1f}%")

with col2:
    input_data = np.array([[facebook, google, influencer, tv, promo]])
    prediction = model.predict(input_data)
    predicted_orders = prediction[0]
    roi = predicted_orders / current_total if current_total > 0 else 0

    col2a, col2b, col2c = st.columns(3)

    with col2a:
        st.metric("Predicted Orders", f"{predicted_orders:.0f}", f"{predicted_orders - 300:.0f}")

    with col2b:
        st.metric("ROI", f"{roi:.2f}x", f"{roi - 1.2:.2f}x")

    with col2c:
        st.metric("Total Spend", f"₹{current_total}k", f"{current_total - 200}k")

    if current_total > 0:
        fig_pie = px.pie(
            values=[facebook, google, influencer, tv],
            names=["Facebook", "Google", "Influencer", "TV"],
            title="Spend Allocation"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

# SHAP & Efficiency
if st.checkbox("Show Advanced Analysis"):
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("SHAP Feature Importance")
        try:
            import shap
            explainer = shap.Explainer(model)
            shap_values = explainer(input_data)
            shap.summary_plot(shap_values, input_data, 
                              feature_names=['Facebook', 'Google', 'Influencer', 'TV', 'Promo'], 
                              show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        except:
            st.info("SHAP not available. Showing mock importances.")
            st.bar_chart([0.4, 0.25, 0.15, 0.15, 0.05])

    with col4:
        st.subheader("Spend Efficiency")
        weights = [0.4, 0.25, 0.15, 0.15, 0.05]
        efficiency = [
            (predicted_orders * weights[0]) / facebook if facebook > 0 else 0,
            (predicted_orders * weights[1]) / google if google > 0 else 0,
            (predicted_orders * weights[2]) / influencer if influencer > 0 else 0,
            (predicted_orders * weights[3]) / tv if tv > 0 else 0
        ]
        df_eff = pd.DataFrame({
            'Channel': ['Facebook', 'Google', 'Influencer', 'TV'],
            'Efficiency': efficiency
        })
        fig_eff = px.bar(df_eff, x='Channel', y='Efficiency', color='Efficiency',
                         title="Orders per ₹ Spent", color_continuous_scale='Viridis')
        st.plotly_chart(fig_eff, use_container_width=True)

# Scenario Comparison
st.subheader("Scenario Comparison")
scenarios = {
    "Current": [facebook, google, influencer, tv, promo],
    "Digital Heavy": [0.4*current_total, 0.4*current_total, 0.1*current_total, 0.1*current_total, promo],
    "TV Heavy": [0.2*current_total, 0.2*current_total, 0.1*current_total, 0.5*current_total, promo],
    "Even Split": [0.25*current_total]*4 + [promo]
}

results = []
for name, alloc in scenarios.items():
    pred = model.predict(np.array([alloc]))[0]
    roi_val = pred / sum(alloc[:4]) if sum(alloc[:4]) > 0 else 0
    results.append({
        "Scenario": name,
        "Facebook": alloc[0], "Google": alloc[1], "Influencer": alloc[2], "TV": alloc[3],
        "Promo %": alloc[4], "Predicted Orders": pred, "ROI": roi_val
    })

df_scenarios = pd.DataFrame(results).round(2)
st.dataframe(df_scenarios, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🔍 **Tip:** Use the sliders to test different combinations and track your ROI and expected Zomato orders.")
