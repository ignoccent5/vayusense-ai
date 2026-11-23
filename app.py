import os
import re
from typing import Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_TIMEOUT_SECONDS = 10
CACHE_TTL_SECONDS = 600
MAX_CITY_LENGTH = 100
DEFAULT_MODEL = "gemini-2.5-flash"

AQI_THRESHOLDS: list[tuple[float, str, str, str]] = [
    (50.0, "Good", "🟢", "Air quality is satisfactory with minimal health risk for outdoor activities."),
    (100.0, "Moderate", "🟡", "Acceptable air quality; unusually sensitive individuals should limit prolonged outdoor exertion."),
    (150.0, "Unhealthy for Sensitive Groups", "🟠", "Sensitive groups (asthma, children, elderly, outdoor athletes) should reduce prolonged outdoor effort."),
    (200.0, "Unhealthy", "🔴", "Everyone may begin experiencing health effects; limit outdoor exertion and keep indoor air clean."),
    (300.0, "Very Unhealthy", "🟣", "Health alert: avoid prolonged outdoor exposure, use an N95/FFP2 mask if going out, and run air filtration."),
    (float("inf"), "Hazardous", "🟤", "Emergency health warning: stay indoors with sealed windows and clean air whenever possible."),
]


def sanitize_city_name(city_name: str) -> str:
    """Validate and normalize a user-supplied city name safely."""
    if not isinstance(city_name, str):
        raise ValueError("City name must be a text string.")
    cleaned = " ".join(city_name.strip().split())
    if not cleaned:
        raise ValueError("Please enter a valid city name.")
    if len(cleaned) > MAX_CITY_LENGTH:
        raise ValueError(f"City name must be {MAX_CITY_LENGTH} characters or fewer.")
    if not re.fullmatch(r"[A-Za-zÀ-ÿ0-9\s,.'()-]+", cleaned):
        raise ValueError("City name contains unsupported characters.")
    return cleaned


def sanitize_prompt_field(value: str, max_length: int = 120) -> str:
    """Sanitize profile/commute/goal text before embedding in an LLM prompt."""
    cleaned = " ".join(str(value).replace("\n", " ").replace("\r", " ").split())
    return cleaned[:max_length]


def build_http_session() -> requests.Session:
    """Create a reusable HTTP session with bounded retries."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_aqi_badge(aqi: float | int | None) -> tuple[str, str]:
    """Return status label with indicator and guidance for a US AQI value."""
    if aqi is None or isinstance(aqi, bool):
        return "⚪ Unknown", "Air quality index data is currently unavailable for this location."
    try:
        numeric_aqi = max(0.0, float(aqi))
    except (TypeError, ValueError):
        return "⚪ Unknown", "Air quality index value could not be interpreted."

    for upper_bound, label, icon, guidance in AQI_THRESHOLDS:
        if numeric_aqi <= upper_bound:
            return f"{icon} {label}", guidance
    return "🟤 Hazardous", AQI_THRESHOLDS[-1][3]


def fetch_city_coordinates(city_name: str, session: requests.Session | None = None) -> dict[str, Any] | None:
    """Fetch latitude, longitude, and display name from Open-Meteo Geocoding API."""
    valid_city = sanitize_city_name(city_name)
    client = session or build_http_session()
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = client.get(
        url,
        params={"name": valid_city, "count": 1, "language": "en", "format": "json"},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results") if isinstance(payload, dict) else None
    if not results:
        return None
    top = results[0]
    return {
        "name": str(top.get("name", valid_city)),
        "country": str(top.get("country", "")),
        "lat": float(top["latitude"]),
        "lon": float(top["longitude"]),
    }


def fetch_air_quality(lat: float, lon: float, session: requests.Session | None = None) -> dict[str, Any]:
    """Fetch current AQI and pollutant readings from Open-Meteo Air Quality API."""
    if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
        raise ValueError("Coordinates are out of valid geographic bounds.")
    client = session or build_http_session()
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "current": "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide",
        "timezone": "auto",
    }
    resp = client.get(url, params=params, timeout=DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    payload = resp.json()
    current = payload.get("current") if isinstance(payload, dict) else None
    return current if isinstance(current, dict) else {}


def build_plan_prompt(city: str, aq_data: dict[str, Any], profile: str, commute: str, goal: str) -> str:
    """Build a structured prompt for generating personalized health & emission guidance."""
    safe_city = sanitize_prompt_field(city)
    safe_profile = sanitize_prompt_field(profile)
    safe_commute = sanitize_prompt_field(commute)
    safe_goal = sanitize_prompt_field(goal)

    return (
        "You are VayuSense AI, an expert Urban Air Quality & Sustainability Assistant.\n"
        f"Location: {safe_city}\n"
        "Real-Time Air Quality Metrics:\n"
        f"- US AQI: {aq_data.get('us_aqi', 'N/A')}\n"
        f"- PM2.5: {aq_data.get('pm2_5', 'N/A')} ug/m3\n"
        f"- PM10: {aq_data.get('pm10', 'N/A')} ug/m3\n"
        f"- NO2: {aq_data.get('nitrogen_dioxide', 'N/A')} ug/m3\n"
        f"- Ozone (O3): {aq_data.get('ozone', 'N/A')} ug/m3\n\n"
        "User Context:\n"
        f"- Health / Activity Profile: {safe_profile}\n"
        f"- Primary Daily Commute: {safe_commute}\n"
        f"- Main Goal: {safe_goal}\n\n"
        "Provide a clear, practical response with these 3 sections:\n"
        "1. Immediate Health & Outdoor Schedule\n"
        "2. Indoor Air Quality & Ventilation Hacks\n"
        "3. Low-Emission Commute & Impact Action\n"
    )


def generate_ai_plan(
    city: str,
    aq_data: dict[str, Any],
    profile: str,
    commute: str,
    goal: str,
    api_key: str,
    genai_client: Any | None = None,
) -> str:
    """Generate a personalized air-quality & low-emission action plan via Gemini."""
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError("A valid Gemini API key is required.")
    client = genai_client or genai.Client(api_key=cleaned_key)
    prompt = build_plan_prompt(city, aq_data, profile, commute, goal)
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Empty response received from Gemini model.")
    return text


def get_default_api_key() -> str:
    """Resolve Gemini API key from Streamlit secrets or environment without exposing secrets."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if isinstance(secret_key, str) and secret_key.strip():
            return secret_key.strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


@st.cache_resource
def _cached_http_session() -> requests.Session:
    return build_http_session()


@st.cache_resource
def _cached_genai_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached_city_coordinates(city_name: str) -> dict[str, Any] | None:
    return fetch_city_coordinates(city_name, session=_cached_http_session())


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached_air_quality(lat: float, lon: float) -> dict[str, Any]:
    return fetch_air_quality(lat, lon, session=_cached_http_session())


def _render_live_status(message: str, level: str = "info") -> None:
    """Render an accessible status banner announced to assistive technologies."""
    aria_html = (
        f'<div role="status" aria-live="polite" '
        f'style="position:relative;padding:0.65rem 0.9rem;border-radius:0.5rem;'
        f'background:#eef6ff;color:#0b2545;border-left:4px solid #0b5ed7;margin:0.5rem 0;">'
        f"{message}</div>"
    )
    if level == "error":
        st.error(message)
    elif level == "warning":
        st.warning(message)
    elif level == "success":
        st.success(message)
    else:
        st.info(message)
    st.markdown(aria_html, unsafe_allow_html=True)


def render_app() -> None:
    """Render the VayuSense AI Streamlit user interface."""
    st.set_page_config(
        page_title="VayuSense AI | Urban Air Quality Assistant",
        page_icon="🌬️",
        layout="wide",
    )

    st.title("🌬️ VayuSense AI — Urban Air Quality Assistant")
    st.caption(
        "Real-time hyperlocal air quality monitoring paired with personalized health and low-emission guidance."
    )

    with st.sidebar:
        st.header("Configuration & Profile")
        default_key = get_default_api_key()
        api_key = st.text_input(
            "Gemini API Key",
            value=default_key,
            type="password",
            help="Enter your Gemini API key or set GEMINI_API_KEY in environment/secrets.",
        )
        city_input = st.text_input(
            "Target City",
            value="New Delhi",
            max_chars=MAX_CITY_LENGTH,
            help="Enter a city name such as New Delhi, Mumbai, or Bengaluru.",
        )
        profile = st.selectbox(
            "Health & Activity Profile",
            [
                "General Adult",
                "Asthma / Respiratory Sensitivity",
                "Outdoor Runner / Cyclist",
                "Parent with Young Children",
                "Elderly",
            ],
            help="Select your primary health or outdoor activity profile.",
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
            help="Select how you travel most days.",
        )
        goal = st.selectbox(
            "Primary Sustainability & Health Goal",
            [
                "Minimize Daily Pollution Exposure",
                "Reduce Personal Carbon & NO2 Footprint",
                "Safe Outdoor Workout Planning",
                "Improve Home Indoor Air Quality",
            ],
            help="Choose your main objective for today's plan.",
        )
        analyze_clicked = st.button(
            "Analyze Air Quality & Generate Plan",
            type="primary",
            use_container_width=True,
        )

    if not analyze_clicked:
        _render_live_status(
            "Ready: Enter a city and Gemini API key in the sidebar, then select Analyze Air Quality & Generate Plan.",
            level="info",
        )
        return

    if not api_key.strip():
        _render_live_status(
            "Missing API key: Please provide a Gemini API key in the sidebar or configure GEMINI_API_KEY.",
            level="error",
        )
        return

    try:
        clean_city = sanitize_city_name(city_input)
    except ValueError as exc:
        _render_live_status(str(exc), level="warning")
        return

    try:
        with st.spinner(f"Fetching air quality data for {clean_city}..."):
            location = _cached_city_coordinates(clean_city)
            if not location:
                _render_live_status(
                    f"No matching location found for '{clean_city}'. Try a major city name.",
                    level="warning",
                )
                return
            aq_data = _cached_air_quality(location["lat"], location["lon"])
    except requests.RequestException:
        _render_live_status(
            "Unable to reach the air quality service right now. Please try again shortly.",
            level="error",
        )
        return

    aqi_val = aq_data.get("us_aqi", 0)
    badge, guidance = get_aqi_badge(aqi_val)

    st.subheader(f"Air Quality Summary — {location['name']} ({location['country']})")
    _render_live_status(f"Current status: {badge}. {guidance}", level="info")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(label="US Air Quality Index", value=f"{aqi_val}")
    col2.metric(label="PM2.5 Fine Particles", value=f"{aq_data.get('pm2_5', 'N/A')} µg/m³")
    col3.metric(label="PM10 Coarse Particles", value=f"{aq_data.get('pm10', 'N/A')} µg/m³")
    col4.metric(label="Nitrogen Dioxide (NO2)", value=f"{aq_data.get('nitrogen_dioxide', 'N/A')} µg/m³")
    col5.metric(label="Ozone (O3)", value=f"{aq_data.get('ozone', 'N/A')} µg/m³")

    st.divider()
    st.subheader("Personalized Action Plan")
    try:
        with st.spinner("Generating your personalized health and low-emission routine..."):
            client = _cached_genai_client(api_key.strip())
            plan_text = generate_ai_plan(
                f"{location['name']}, {location['country']}",
                aq_data,
                profile,
                commute,
                goal,
                api_key.strip(),
                genai_client=client,
            )
        st.markdown(plan_text)
        _render_live_status("Personalized action plan generated.", level="success")
    except Exception:
        _render_live_status(
            "Could not generate the AI action plan. Please verify your API key and try again.",
            level="error",
        )


if __name__ == "__main__":
    render_app()
