import streamlit as st
import pandas as pd
import numpy as np
from predict import TsunamiPredictor
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="Tsunami Prediction", page_icon="🌊", layout="wide")

# Major cities and earthquake-prone locations with coordinates
MAJOR_LOCATIONS = {
    "Custom Location": (None, None),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Los Angeles, USA": (34.0522, -118.2437),
    "San Francisco, USA": (37.7749, -122.4194),
    "Manila, Philippines": (14.5995, 120.9842),
    "Jakarta, Indonesia": (-6.2088, 106.8456),
    "Wellington, New Zealand": (-41.2865, 174.7762),
    "Anchorage, Alaska": (61.2181, -149.9003),
    "Vancouver, Canada": (49.2827, -123.1207),
    "Athens, Greece": (37.9838, 23.7275),
    "Mexico City, Mexico": (19.4326, -99.1332),
    "Pune, India": (18.5246, 73.8786),
}

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

    # Location selection method
    st.markdown("### 📍 Choose Location Input Method")
    location_method = st.radio(
        "How would you like to set the location?",
        ["Select from Major Cities", "Use My Current Location", "Enter Manually"],
        horizontal=True,
        help="Choose a pre-defined city, use your device's GPS, or enter coordinates manually",
    )

    # Initialize session state for location
    if "user_latitude" not in st.session_state:
        st.session_state.user_latitude = 35.0
    if "user_longitude" not in st.session_state:
        st.session_state.user_longitude = 140.0

    latitude = st.session_state.user_latitude
    longitude = st.session_state.user_longitude

    if location_method == "Select from Major Cities":
        selected_city = st.selectbox(
            "🌆 Select a Major City or Earthquake-Prone Location",
            options=list(MAJOR_LOCATIONS.keys())[1:],  # Exclude "Custom Location"
            help="Choose from major earthquake-prone cities around the world",
        )
        if selected_city:
            latitude, longitude = MAJOR_LOCATIONS[selected_city]
            st.session_state.user_latitude = latitude
            st.session_state.user_longitude = longitude
            st.success(
                f"📍 Location set to: **{selected_city}** ({latitude:.4f}, {longitude:.4f})"
            )

    elif location_method == "Use My Current Location":
        st.info("🌐 Click the button below to get your current location")

        if st.button("📍 Get My Location", type="primary"):
            location = streamlit_geolocation()

            if location and location.get("latitude") and location.get("longitude"):
                st.session_state.user_latitude = location["latitude"]
                st.session_state.user_longitude = location["longitude"]
                latitude = location["latitude"]
                longitude = location["longitude"]

                st.success(f"✅ Location retrieved successfully!")
                st.write(f"**Latitude:** {latitude:.6f}")
                st.write(f"**Longitude:** {longitude:.6f}")

                # Display location on map
                st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}))
            else:
                st.warning(
                    "⚠️ No location information available. Please grant location permission in your browser."
                )
                st.info(
                    "💡 If location access is blocked, you can enter coordinates manually below."
                )

        st.markdown("---")
        st.markdown("**Or enter your coordinates manually:**")
        col_loc1, col_loc2 = st.columns(2)
        with col_loc1:
            manual_lat = st.number_input(
                "Your Latitude",
                value=st.session_state.user_latitude,
                min_value=-90.0,
                max_value=90.0,
                step=0.0001,
                format="%.4f",
                key="manual_lat",
            )
        with col_loc2:
            manual_lon = st.number_input(
                "Your Longitude",
                value=st.session_state.user_longitude,
                min_value=-180.0,
                max_value=180.0,
                step=0.0001,
                format="%.4f",
                key="manual_lon",
            )

        st.session_state.user_latitude = manual_lat
        st.session_state.user_longitude = manual_lon
        latitude = manual_lat
        longitude = manual_lon

        st.caption(
            "💡 Tip: You can find your coordinates using Google Maps - right-click on your location and select the coordinates."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Earthquake Epicenter")

        if location_method == "Enter Manually":
            latitude = st.number_input(
                "Latitude",
                value=st.session_state.user_latitude,
                min_value=-90.0,
                max_value=90.0,
                step=0.1,
                help="Latitude of earthquake epicenter",
                key="custom_lat",
            )
            longitude = st.number_input(
                "Longitude",
                value=st.session_state.user_longitude,
                min_value=-180.0,
                max_value=180.0,
                step=0.1,
                help="Longitude of earthquake epicenter",
                key="custom_lon",
            )
            st.session_state.user_latitude = latitude
            st.session_state.user_longitude = longitude
        else:
            # Display selected location
            st.metric("Latitude", f"{latitude:.4f}°")
            st.metric("Longitude", f"{longitude:.4f}°")

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
