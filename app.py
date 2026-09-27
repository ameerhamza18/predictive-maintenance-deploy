"""
app.py
Streamlit interface for the predictive maintenance failure classifier.
Deployed on Streamlit Community Cloud.
"""
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

from data_prep import engineer_features
from explain import get_feature_names

THRESHOLD = 0.5

st.set_page_config(page_title="Predictive Maintenance: Machine Failure Risk", page_icon="🏭")


@st.cache_resource
def load_model():
    model = joblib.load("xgb_churn_model.pkl")
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    feature_names = get_feature_names(preprocessor)
    return model, preprocessor, explainer, feature_names


model, preprocessor, explainer, feature_names = load_model()

st.title("🏭 Predictive Maintenance: Machine Failure Risk")
st.markdown(
    "Enter live sensor readings to get a failure-risk prediction, explained via SHAP.\n\n"
    "Trained on the AI4I 2020 UCI dataset (10,000 machines, 3.4% failure rate). "
    "[Source code](https://github.com/YOUR_USERNAME/predictive-maintenance-ml)"
)

col1, col2 = st.columns(2)

with col1:
    machine_type = st.radio("Product Quality Variant", ["L", "M", "H"], index=1, horizontal=True)
    air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0)
    process_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0)
    rot_speed = st.slider("Rotational Speed (rpm)", 1150, 2900, 1500)
    torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0)
    tool_wear = st.slider("Tool Wear (min)", 0, 253, 100)

    st.markdown("**Try an example:**")
    example = st.selectbox(
        "Scenario",
        ["Custom (use sliders above)", "Normal operation", "High wear + high torque", "Small temp differential"],
        label_visibility="collapsed",
    )

EXAMPLES = {
    "Normal operation": ("L", 298.1, 308.6, 1551, 42.8, 0),
    "High wear + high torque": ("L", 300.9, 311.2, 1350, 65.0, 220),
    "Small temp differential": ("M", 300.5, 302.0, 1400, 30.0, 100),
}

if example in EXAMPLES:
    machine_type, air_temp, process_temp, rot_speed, torque, tool_wear = EXAMPLES[example]

raw = pd.DataFrame([{
    "type": machine_type,
    "air_temp": air_temp,
    "process_temp": process_temp,
    "rot_speed": rot_speed,
    "torque": torque,
    "tool_wear": tool_wear,
}])
X_row = engineer_features(raw).drop(columns=["overstrain_threshold"])

proba = model.predict_proba(X_row)[0, 1]
X_transformed = preprocessor.transform(X_row)
shap_values_row = explainer.shap_values(X_transformed)

with col2:
    if proba >= THRESHOLD:
        st.error(f"⚠️ FAILURE RISK — probability: {proba:.1%}")
    else:
        st.success(f"✅ Normal operation — failure probability: {proba:.1%}")

    exp = shap.Explanation(
        values=shap_values_row[0],
        base_values=explainer.expected_value,
        data=X_transformed[0],
        feature_names=feature_names,
    )
    fig, ax = plt.subplots()
    shap.plots.waterfall(exp, show=False, max_display=10)
    st.pyplot(plt.gcf())