import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost  # Required for loading the models
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components # <<< IMPORT THE COMPONENTS LIBRARY

# --- Configuration & Data Loading ---

st.set_page_config(
    page_title="Insurity PremiumSense Dashboard",
    page_icon="💡",
    layout="wide"
)

# Use Streamlit's cache to load data once and speed up the app
@st.cache_data
def load_data_and_models():
    """
    Loads all necessary data and models.
    Returns them in a dictionary for easy access.
    """
    try:
        premium_model = joblib.load('premium_model.joblib')
        premium_model_features = joblib.load('premium_model_features.joblib')
        telematics_df = pd.read_csv('model_ready_features.csv')
        premium_df = pd.read_csv('premium_features.csv')
        full_df = pd.merge(premium_df, telematics_df.drop(columns=['profile_name']), on='driver_id', how='left')

        # Pre-calculate the SHAP explainer and values
        explainer = shap.TreeExplainer(premium_model)
        shap_values_summary = explainer(full_df[premium_model_features])

        return {
            "premium_model": premium_model,
            "premium_features": premium_model_features,
            "full_df": full_df,
            "explainer": explainer,
            "shap_values_summary": shap_values_summary
        }
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A required file is missing: {e.filename}")
        st.info("Please ensure you have run all previous scripts successfully.")
        st.stop()

# Load all assets
assets = load_data_and_models()
full_df = assets["full_df"]
premium_model = assets["premium_model"]
premium_model_features = assets["premium_features"]
explainer = assets["explainer"]
shap_values_summary = assets["shap_values_summary"]

# --- Main Application Logic ---

st.title("💡 Insurity PremiumSense Dashboard")

st.sidebar.title("Policy Holder")
driver_list = sorted(full_df['driver_id'].unique())
selected_driver = st.sidebar.selectbox("Select a Driver to Analyze", driver_list)

driver_data = full_df[full_df['driver_id'] == selected_driver].iloc[0]

view_mode = st.radio(
    "Select Dashboard View",
    ["👤 Customer View", "🏢 Underwriter View"],
    horizontal=True,
    label_visibility="collapsed"
)

behavioral_risk_score = driver_data['behavioral_risk_score']
predicted_annual_loss = premium_model.predict(pd.DataFrame([driver_data[premium_model_features]]))[0]
PROFIT_MARGIN = 0.15
final_annual_premium = predicted_annual_loss * (1 + PROFIT_MARGIN)


# --- 👤 CUSTOMER VIEW (Unchanged) ---
if view_mode == "👤 Customer View":
    st.header(f"Welcome, {selected_driver}!")
    st.markdown("Here’s a simple breakdown of your insurance premium and how your driving habits affect it.")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Your Personalized Annual Premium")
        st.markdown(f"<h1 style='color: #4CAF50; font-size: 48px;'>${final_annual_premium:,.2f}</h1>", unsafe_allow_html=True)
    with col2:
        st.metric("Equivalent Monthly Cost", f"${final_annual_premium/12:,.2f}")
    st.markdown("---")
    st.subheader("What Determines Your Premium?")
    col1, col2 = st.columns(2)
    with col1:
        st.info("#### Your Driving Score")
        st.progress(int(behavioral_risk_score), text=f"Score: {behavioral_risk_score:.1f} / 100")
        if behavioral_risk_score >= 80:
            st.markdown("✅ **Excellent!** Your safe driving habits are significantly lowering your premium.")
        elif behavioral_risk_score >= 60:
            st.markdown("👍 **Good.** Consistent driving is helping you maintain a fair premium.")
        else:
            st.markdown("⚠️ **Needs Improvement.** Certain driving habits are increasing your premium.")
    with col2:
        st.info("#### Other Key Factors")
        st.markdown(f"- **Vehicle Age:** `{driver_data['vehicle_age_years']}` years")
        st.markdown(f"- **Location Risk:** `{'High' if driver_data['theft_rate'] > 1.5 else 'Medium'}`")
        st.markdown(f"- **Annual Mileage:** `{driver_data['annual_mileage']:,}` miles")
    st.markdown("---")
    st.subheader("How You Can Save Money 💰")
    if driver_data['accels_per_100km'] > 50 or driver_data['brakes_per_100km'] > 50:
        st.success("**Top Tip:** Try smoother acceleration and braking. This is the fastest way to improve your Driving Score and lower your premium.")
    if driver_data['late_night_driving_percentage'] > 10:
        st.success("**Top Tip:** Reducing late-night trips can lower your risk profile and your premium.")
    st.success("**Long-Term:** A well-maintained vehicle with a good servicing history also helps keep costs down.")

# --- 🏢 UNDERWRITER VIEW (Revised and Fixed) ---
elif view_mode == "🏢 Underwriter View":
    st.header(f"Underwriting Analysis: `{selected_driver}`")
    st.markdown("Detailed breakdown of predicted loss and contributing risk factors based on the Optuna-optimized pricing model.")

    st.markdown("---")
    st.subheader("Financial Breakdown")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Annual Loss", f"${predicted_annual_loss:,.2f}", help="The model's prediction of total claims cost for this policyholder over one year.")
    col2.metric("Profit Margin (15%)", f"${predicted_annual_loss * PROFIT_MARGIN:,.2f}")
    col3.metric("Final Quoted Premium", f"${final_annual_premium:,.2f}", "Predicted Loss + Margin")
    st.markdown("---")

    st.subheader("1. Overall Model Feature Importance")
    st.markdown("This chart shows which factors have the biggest impact on the premium prediction **across all drivers**. This gives context to the model's general behavior.")
    fig_summary, ax_summary = plt.subplots()
    shap.summary_plot(shap_values_summary, full_df[premium_model_features], plot_type="bar", show=False)
    st.pyplot(fig_summary)
    plt.close()
    st.markdown("---")

    # <<< THIS IS THE CORRECTED AND ROBUST METHOD FOR THE FORCE PLOT >>>
    st.subheader(f"2. Specific Analysis for `{selected_driver}`")
    st.markdown("This force plot shows the precise push-and-pull of factors for this **individual driver's** premium. Red factors increased the predicted loss, blue factors decreased it.")

    def st_shap(plot, height=None):
        """
        A wrapper function to display a SHAP plot in Streamlit.
        """
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    # Get the index of the selected driver to find their specific SHAP values
    driver_index = full_df[full_df['driver_id'] == selected_driver].index[0]
    
    # Generate the plot object and then pass it to our wrapper function
    force_plot = shap.plots.force(shap_values_summary[driver_index])
    st_shap(force_plot, height=150)
    # <<< END OF CORRECTION >>>
    
    st.markdown("---")
    st.subheader("3. Full Factor Drill-Down")
    with st.expander("Click to view all input features for the pricing model"):
        st.dataframe(pd.DataFrame(driver_data[premium_model_features]).rename(columns={driver_data.name: 'Value'}))