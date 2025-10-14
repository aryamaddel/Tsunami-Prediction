import streamlit as st
import pandas as pd
import numpy as np
from predict import TsunamiPredictor
from datetime import datetime

st.set_page_config(page_title="Tsunami Prediction", page_icon="🌊", layout="wide")

st.title("🌊 Tsunami Prediction System")
st.markdown(
    "Predict tsunami severity and intensity based on earthquake and geological data"
)


@st.cache_resource
def load_model():
    return TsunamiPredictor("tsunami_models.pth")


try:
    predictor = load_model()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location Data")
        latitude = st.number_input(
            "Latitude", value=35.0, min_value=-90.0, max_value=90.0, step=0.1
        )
        longitude = st.number_input(
            "Longitude", value=140.0, min_value=-180.0, max_value=180.0, step=0.1
        )

    with col2:
        st.subheader("🌍 Earthquake Data")
        earthquake_magnitude = st.number_input(
            "Earthquake Magnitude", value=7.0, min_value=0.0, max_value=10.0, step=0.1
        )
        water_height = st.number_input(
            "Maximum Water Height (m)",
            value=5.0,
            min_value=0.0,
            max_value=100.0,
            step=0.5,
        )

    with col3:
        st.subheader("📊 Additional Parameters")
        intensity = st.slider("Tsunami Intensity", 0, 5, 3)
        tsunami_magnitude_abe = st.number_input(
            "Tsunami Magnitude (Abe)",
            value=0.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
        )
        tsunami_magnitude_iida = st.number_input(
            "Tsunami Magnitude (Iida)",
            value=0.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
        )

    with st.expander("⚙️ Advanced Settings (Optional)", expanded=False):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            year = st.number_input(
                "Year", value=datetime.now().year, min_value=1900, max_value=2100
            )
            month = st.number_input(
                "Month", value=datetime.now().month, min_value=1, max_value=12
            )
            day = st.number_input(
                "Day", value=datetime.now().day, min_value=1, max_value=31
            )
        with col_adv2:
            event_validity = st.selectbox("Event Validity", [1, 2, 3, 4], index=3)
            cause_code = st.selectbox("Cause Code", [1, 2, 3, 4], index=0)
            num_runups = st.number_input(
                "Number of Runups", value=1, min_value=0, max_value=100
            )

    if st.button("🔮 Predict Tsunami", type="primary", use_container_width=True):
        with st.spinner("Analyzing data..."):
            result = predictor.predict(
                earthquake_magnitude=earthquake_magnitude,
                latitude=latitude,
                longitude=longitude,
                water_height=water_height,
                tsunami_magnitude_abe=(
                    tsunami_magnitude_abe if tsunami_magnitude_abe > 0 else None
                ),
                tsunami_magnitude_iida=(
                    tsunami_magnitude_iida if tsunami_magnitude_iida > 0 else None
                ),
                intensity=intensity,
                year=year,
                month=month,
                day=day,
                event_validity=event_validity,
                cause_code=cause_code,
                num_runups=num_runups,
            )

        st.success("Prediction Complete!")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric(
                label="Tsunami Severity",
                value=result["severity"],
                delta=result["severity_confidence"],
            )
            severity_color = {
                "Minor": "🟢",
                "Moderate": "🟡",
                "Major": "🟠",
                "Extreme": "🔴",
            }
            st.markdown(
                f"### {severity_color.get(result['severity'], '⚪')} {result['severity']}"
            )

        with col_res2:
            st.metric(
                label="Tsunami Intensity",
                value=result["intensity"],
                delta=result["intensity_confidence"],
            )
            intensity_color = {"Low Intensity": "🟢", "High Intensity": "🔴"}
            st.markdown(
                f"### {intensity_color.get(result['intensity'], '⚪')} {result['intensity']}"
            )

        st.divider()

        st.subheader("📋 Input Summary")
        summary_data = {
            "Parameter": [
                "Earthquake Magnitude",
                "Latitude",
                "Longitude",
                "Water Height (m)",
                "Tsunami Intensity",
            ],
            "Value": [
                earthquake_magnitude,
                latitude,
                longitude,
                water_height,
                intensity,
            ],
        }
        st.dataframe(
            pd.DataFrame(summary_data), use_container_width=True, hide_index=True
        )

    st.divider()
    st.markdown("### 📖 About")
    st.info(
        """
    This system uses a PyTorch neural network trained on historical tsunami data to predict:
    - **Severity**: Minor, Moderate, Major, or Extreme
    - **Intensity**: Low or High
    
    The model considers earthquake magnitude, location, water height, and other geological factors.
    """
    )

    with st.expander("ℹ️ How to Use"):
        st.markdown(
            """
        1. Enter the **location** (latitude and longitude) of the event
        2. Specify **earthquake magnitude** and **water height**
        3. Adjust **tsunami intensity** and other parameters as needed
        4. Click **Predict Tsunami** to get results
        5. Review the predicted severity and intensity with confidence scores
        """
        )

except FileNotFoundError:
    st.error(
        "⚠️ Model file not found! Please train the model first by running `python train.py`"
    )
except Exception as e:
    st.error(f"⚠️ An error occurred: {str(e)}")
    st.info("Please ensure the model is trained and all dependencies are installed.")
