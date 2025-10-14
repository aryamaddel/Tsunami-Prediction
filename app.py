import streamlit as st
import pandas as pd
import numpy as np
from predict import TsunamiPredictor
from datetime import datetime

st.set_page_config(page_title="Tsunami Prediction", page_icon="🌊", layout="wide")

st.title("🌊 Tsunami Early Warning System")
st.markdown(
    "Predict tsunami risk and characteristics based on **earthquake parameters only**"
)
st.info(
    "ℹ️ Enter earthquake details below. The system will predict if a tsunami will occur and its potential severity."
)


@st.cache_resource
def load_model():
    return TsunamiPredictor("tsunami_models.pth")


try:
    predictor = load_model()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Earthquake Epicenter")
        latitude = st.number_input(
            "Latitude",
            value=35.0,
            min_value=-90.0,
            max_value=90.0,
            step=0.1,
            help="Latitude of earthquake epicenter",
        )
        longitude = st.number_input(
            "Longitude",
            value=140.0,
            min_value=-180.0,
            max_value=180.0,
            step=0.1,
            help="Longitude of earthquake epicenter",
        )

    with col2:
        st.subheader("🌍 Earthquake Properties")
        earthquake_magnitude = st.number_input(
            "Earthquake Magnitude (Richter Scale)",
            value=7.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            help="Magnitude of the earthquake on Richter scale",
        )
        earthquake_depth = st.number_input(
            "Earthquake Depth (km)",
            value=10.0,
            min_value=0.0,
            max_value=700.0,
            step=1.0,
            help="Depth of earthquake hypocenter in kilometers",
        )

    with st.expander("⚙️ Advanced Settings (Optional)", expanded=False):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            st.markdown("**Event Timing**")
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
            st.markdown("**Event Classification**")
            cause_code = st.selectbox(
                "Tsunami Cause",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Earthquake",
                    2: "Volcanic Eruption",
                    3: "Landslide",
                    4: "Other",
                }.get(x, "Unknown"),
                index=0,
                help="Primary cause of potential tsunami",
            )
            vol = st.number_input(
                "Volcanic Explosivity Index (VEI)",
                value=0,
                min_value=0,
                max_value=8,
                help="Only relevant if cause is volcanic (0 = not volcanic)",
            )

    if st.button("🔮 Predict Tsunami Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing earthquake data..."):
            result = predictor.predict(
                earthquake_magnitude=earthquake_magnitude,
                latitude=latitude,
                longitude=longitude,
                earthquake_depth=earthquake_depth,
                year=year,
                month=month,
                day=day,
                cause_code=cause_code,
                vol=vol,
            )

        st.success("✅ Prediction Complete!")

        # Display tsunami risk assessment
        st.markdown("---")
        st.markdown("## 🌊 Tsunami Risk Assessment")

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric(
                label="Predicted Tsunami Severity",
                value=result["severity"],
                delta=f"Confidence: {result['severity_confidence']}",
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
                label="Predicted Tsunami Intensity",
                value=result["intensity"],
                delta=f"Confidence: {result['intensity_confidence']}",
            )
            intensity_color = {"Low Intensity": "🟢", "High Intensity": "🔴"}
            st.markdown(
                f"### {intensity_color.get(result['intensity'], '⚪')} {result['intensity']}"
            )

        st.divider()

        # Display warning based on severity
        if result["severity"] in ["Major", "Extreme"]:
            st.error(
                "🚨 **HIGH TSUNAMI RISK DETECTED** - Immediate evacuation recommended for coastal areas!"
            )
        elif result["severity"] == "Moderate":
            st.warning(
                "⚠️ **MODERATE TSUNAMI RISK** - Stay alert and monitor official warnings."
            )
        else:
            st.info("ℹ️ **LOW TSUNAMI RISK** - Remain cautious near coastal areas.")

        st.subheader("📋 Earthquake Input Summary")
        summary_data = {
            "Parameter": [
                "Earthquake Magnitude",
                "Earthquake Depth (km)",
                "Latitude",
                "Longitude",
                "Event Date",
                "Tsunami Cause",
            ],
            "Value": [
                earthquake_magnitude,
                earthquake_depth,
                latitude,
                longitude,
                f"{year}-{month:02d}-{day:02d}",
                {1: "Earthquake", 2: "Volcanic", 3: "Landslide", 4: "Other"}.get(
                    cause_code, "Unknown"
                ),
            ],
        }
        st.dataframe(
            pd.DataFrame(summary_data), use_container_width=True, hide_index=True
        )

    st.divider()
    st.markdown("### 📖 About This System")
    st.info(
        """
    This **Tsunami Early Warning System** uses a PyTorch neural network trained on historical tsunami data to predict:
    - **Tsunami Occurrence Risk**: Based solely on earthquake parameters
    - **Severity**: Minor, Moderate, Major, or Extreme
    - **Intensity**: Low or High
    
    **How it works**: The model analyzes earthquake characteristics (magnitude, location, depth) to predict 
    whether a tsunami will occur and its potential impact - **without needing to know tsunami parameters in advance**.
    """
    )

    with st.expander("ℹ️ How to Use This System"):
        st.markdown(
            """
        ### Quick Start Guide
        
        1. **Enter Earthquake Details:**
           - Location (latitude and longitude) of the earthquake epicenter
           - Magnitude on the Richter scale
           - Depth of the earthquake (optional, defaults to 10 km)
        
        2. **Optional Settings:**
           - Event date and time
           - Cause type (earthquake, volcanic, landslide, etc.)
           - Volcanic explosivity index (if volcanic)
        
        3. **Get Predictions:**
           - Click "Predict Tsunami Risk"
           - Review predicted severity and intensity
           - Follow safety recommendations
        
        ### Understanding Results
        
        **Severity Levels:**
        - 🟢 **Minor**: Limited coastal impact
        - 🟡 **Moderate**: Significant coastal flooding possible
        - 🟠 **Major**: Extensive damage to coastal areas
        - 🔴 **Extreme**: Catastrophic tsunami event
        
        **Intensity:**
        - 🟢 **Low**: Wave heights typically < 2 meters
        - 🔴 **High**: Wave heights typically > 2 meters
        
        ### Important Notes
        
        ⚠️ This is a prediction model and should be used alongside official tsunami warning systems.
        Always follow local emergency management guidance.
        """
        )

except FileNotFoundError:
    st.error(
        "⚠️ Model file not found! Please train the model first by running `python train.py`"
    )
except Exception as e:
    st.error(f"⚠️ An error occurred: {str(e)}")
    st.info("Please ensure the model is trained and all dependencies are installed.")
