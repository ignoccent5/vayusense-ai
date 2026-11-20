import os
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(
    page_title="VayuSense AI | Urban Air Quality Assistant",
    page_icon="🌬️",
    layout="wide",
)

MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]


# --- Helper Functions ---
def get_city_coordinates(city_name: str):
    """Fetch latitude and longitude using Open-Meteo's free Geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(
        url,
        params={"name": city_name, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        return None
    top = data["results"][0]
    return {
        "name": top.get("name"),
        "country": top.get("country", ""),
        "lat": top.get("latitude"),
        "lon": top.get("longitude"),
    }


def get_air_quality(lat: float, lon: float):
    """Fetch real-time AQI and pollutants from Open-Meteo Air Quality API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide",
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("current", {})


def get_aqi_badge(aqi: float):
    if aqi is None:
        return "⚪ Unknown", "Air quality index unavailable for this location."
    if aqi <= 50:
        return "🟢 Good", "Safe for all outdoor activities."
    elif aqi <= 100:
        return "🟡 Moderate", "Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "🟠 Unhealthy for Sensitive Groups", "Wear a mask outdoors if you have respiratory conditions."
    elif aqi <= 200:
        return "🔴 Unhealthy", "Avoid heavy outdoor exercise; keep windows closed."
    elif aqi <= 300:
        return "🟣 Very Unhealthy", "Stay indoors with air purification; N95 mask required outside."
    else:
        return "🟤 Hazardous", "Health emergency warning. Avoid all outdoor exposure."


def generate_ai_plan(city: str, aq_data: dict, profile: str, commute: str, goal: str, api_key: str):
    """Use Google Gemini to generate a personalized low-emission & health protection plan."""
    client = genai.Client(api_key=api_key.strip())
    prompt = f"""
    You are VayuSense AI, an expert Urban Air Quality & Sustainability Assistant.

    Location: {city}
    Real-Time Air Quality Metrics:
    - US AQI: {aq_data.get('us_aqi', 'N/A')}
    - PM2.5: {aq_data.get('pm2_5', 'N/A')} µg/m³
    - PM10: {aq_data.get('pm10', 'N/A')} µg/m³
    - NO2 (Traffic/Industrial gas): {aq_data.get('nitrogen_dioxide', 'N/A')} µg/m³
    - Ozone (O3): {aq_data.get('ozone', 'N/A')} µg/m³

    User Context:
    - Health / Activity Profile: {profile}
    - Primary Daily Commute: {commute}
    - Main Goal: {goal}

    Provide a clear, practical, structured response with these 3 sections:
    1. **Immediate Health & Outdoor Schedule**: Exact advice on when to go outside, mask guidance (N95 vs cloth), and exercise adjustments today.
    2. **Indoor Air Quality & Ventilation Hacks**: When to open/close windows based on current pollutants and low-cost indoor air purification tips.
    3. **Low-Emission Commute & Impact Action**: Specific commute alternatives for their mode of transport and how much personal exposure + urban CO2/NO2 it reduces.
    Keep it concise, empathetic, and actionable.
    """

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if response and response.text:
                return response.text, model_name
        except Exception as e:
            last_error = e

    raise RuntimeError(f"All configured Gemini models failed ({', '.join(MODELS_TO_TRY)}): {last_error}")


# --- UI Layout ---
st.title("🌬️ VayuSense AI — Urban Air Quality Assistant")
st.markdown("Real-time hyperlocal pollution tracking paired with **Gemini AI** personalized health & low-emission routines.")

with st.sidebar:
    st.header("⚙️ Configuration")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Get a free key from Google AI Studio",
    )
    st.divider()
    st.header("👤 Your Profile")
    city = st.text_input("City Name", value="New Delhi")
    profile = st.selectbox(
        "Health & Lifestyle Profile",
        [
            "General Adult",
            "Asthma / Respiratory Sensitivity",
            "Outdoor Runner / Cyclist",
            "Parent with Young Children",
            "Elderly",
        ],
    )
    commute = st.selectbox(
        "Primary Daily Commute",
        [
            "Private Car / Cab",
            "Two-Wheeler / Bike",
            "Public Metro / Bus",
            "Walking / Cycling",
            "Work From Home",
        ],
    )
    goal = st.selectbox(
        "Primary Goal",
        [
            "Minimize Daily Pollution Exposure",
            "Reduce Personal Carbon & NO2 Footprint",
            "Safe Outdoor Workout Planning",
            "Improve Home Indoor Air Quality",
        ],
    )
    analyze_btn = st.button("🔍 Analyze Air & Generate Plan", type="primary", use_container_width=True)

if analyze_btn:
    if not api_key or not api_key.strip():
        st.error("Please enter your Gemini API Key in the sidebar (or configure `GEMINI_API_KEY` in app secrets).")
    elif not city.strip():
        st.warning("Please enter a valid city name.")
    else:
        try:
            with st.spinner(f"Fetching live atmospheric data for {city}..."):
                loc = get_city_coordinates(city.strip())

            if not loc:
                st.error(f"Could not find coordinates for '{city}'. Try another major city name.")
            else:
                with st.spinner("Loading live air quality metrics..."):
                    aq = get_air_quality(loc["lat"], loc["lon"])

                aqi_val = aq.get("us_aqi")
                badge, summary = get_aqi_badge(aqi_val)

                st.subheader(f"📍 Live Air Quality: {loc['name']}, {loc['country']}")
                st.info(f"**Status:** {badge} — {summary}")

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("US AQI", f"{aqi_val if aqi_val is not None else 'N/A'}")
                col2.metric("PM2.5", f"{aq.get('pm2_5', 'N/A')} µg/m³")
                col3.metric("PM10", f"{aq.get('pm10', 'N/A')} µg/m³")
                col4.metric("NO₂", f"{aq.get('nitrogen_dioxide', 'N/A')} µg/m³")
                col5.metric("Ozone (O₃)", f"{aq.get('ozone', 'N/A')} µg/m³")

                st.divider()
                with st.spinner("Gemini AI is crafting your personalized routine & low-emission plan..."):
                    ai_advice, used_model = generate_ai_plan(
                        f"{loc['name']}, {loc['country']}",
                        aq,
                        profile,
                        commute,
                        goal,
                        api_key,
                    )
                    st.subheader("🤖 Your Personalized AI Action Plan")
                    st.caption(f"Generated using `{used_model}`")
                    st.markdown(ai_advice)
        except requests.RequestException as req_err:
            st.error(f"Network / Air Quality API Error: {req_err}")
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
else:
    st.info("👈 Enter your **City** and **Gemini API Key** in the left sidebar, then click **Analyze Air & Generate Plan**.")
