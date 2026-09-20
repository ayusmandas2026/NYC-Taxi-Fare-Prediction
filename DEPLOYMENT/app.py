import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import tensorflow as tf

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MetroCab NYC | Fare Calculator",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL HUMAN-CENTERED CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    /* Clean subtle header */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 0 1.2rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .brand-tag {
        font-size: 0.8rem;
        font-weight: 600;
        background: #f1f5f9;
        color: #475569;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
    }

    /* Subdued Header Text */
    .section-caption {
        font-size: 0.88rem;
        color: #64748b;
        margin-top: -0.2rem;
        margin-bottom: 1rem;
    }

    /* Primary Action Button (Sleek Slate instead of jarring red) */
    div.stButton > button:first-child {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.4rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1e293b !important;
        color: #fef08a !important;
        border-color: #eab308 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15) !important;
    }

    /* Ride Receipt Card */
    .receipt-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.2rem;
    }
    .receipt-header {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .receipt-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
    }
    .receipt-price {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        margin-top: 0.2rem;
    }
    .receipt-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: #059669;
        background: #ecfdf5;
        padding: 0.35rem 0.65rem;
        border-radius: 6px;
        border: 1px solid #a7f3d0;
    }
    .receipt-divider {
        height: 1px;
        background: #f1f5f9;
        margin: 1rem 0;
    }
    .receipt-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        font-size: 0.88rem;
    }
    .receipt-row span {
        color: #64748b;
    }
    .receipt-row b {
        color: #1e293b;
        font-weight: 600;
    }

    /* Condition pills */
    .condition-pill {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .condition-normal {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .condition-rush {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .condition-night {
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }
    .condition-airport {
        background: #fefce8;
        color: #854d0e;
        border: 1px solid #fde047;
    }

    /* Quick Preset Chips */
    .preset-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PATHS AND ASSETS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "MODELS"
RESULTS_DIR = BASE_DIR / "RESULTS"
DATA_DIR = BASE_DIR / "DATA"

MODEL_PATH = MODEL_DIR / "final_dnn.keras"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
CONFIG_PATH = MODEL_DIR / "feature_config.json"
TEST_DATA_PATH = DATA_DIR / "raw" / "test.csv"

# ============================================================
# CACHED RESOURCE LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)

try:
    model = load_model()
    scaler = load_scaler()
    config = load_config()
    features_list = config.get("features", [
        "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
        "passenger_count", "distance_km", "year", "month", "day", "hour", "day_of_week"
    ])
except Exception as e:
    st.error("System configuration error: model or preprocessor files are missing.")
    st.exception(e)
    st.stop()

# ============================================================
# HUMAN-FRIENDLY NYC LOCATIONS & PRESETS
# ============================================================

NYC_LOCATIONS = {
    "Times Square (Midtown Manhattan)": (40.7580, -73.9855),
    "Grand Central Terminal (Midtown East)": (40.7527, -73.9772),
    "Penn Station / Madison Square Garden": (40.7505, -73.9934),
    "Central Park South (Columbus Circle)": (40.7681, -73.9819),
    "Empire State Building (Midtown)": (40.7484, -73.9857),
    "Wall Street / Financial District": (40.7075, -74.0090),
    "World Trade Center / Oculus": (40.7115, -74.0125),
    "SoHo (Broadway & Spring St)": (40.7233, -74.0000),
    "Greenwich Village (Washington Square)": (40.7308, -73.9973),
    "DUMBO / Brooklyn Bridge Waterfront": (40.7032, -73.9937),
    "Williamsburg (Bedford Avenue)": (40.7170, -73.9575),
    "Barclays Center (Downtown Brooklyn)": (40.6826, -73.9754),
    "JFK International Airport (Terminal 4)": (40.6413, -73.7781),
    "LaGuardia Airport (Terminal B)": (40.7769, -73.8740),
    "Yankee Stadium (The Bronx)": (40.8296, -73.9262),
    "Columbia University (Morningside Heights)": (40.8075, -73.9626),
}

PRESET_ROUTES = {
    "Midtown to JFK Airport": ("Times Square (Midtown Manhattan)", "JFK International Airport (Terminal 4)"),
    "LaGuardia Airport to Midtown": ("LaGuardia Airport (Terminal B)", "Times Square (Midtown Manhattan)"),
    "Central Park to Wall Street": ("Central Park South (Columbus Circle)", "Wall Street / Financial District"),
    "Empire State to DUMBO": ("Empire State Building (Midtown)", "DUMBO / Brooklyn Bridge Waterfront"),
    "Grand Central to Yankee Stadium": ("Grand Central Terminal (Midtown East)", "Yankee Stadium (The Bronx)")
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c

def calculate_duration(distance_km, is_rush_hour):
    """Estimate realistic transit time in NYC traffic."""
    if distance_km > 15:
        speed = 34.0 if is_rush_hour else 44.0
    elif distance_km > 6:
        speed = 20.0 if is_rush_hour else 26.0
    else:
        speed = 13.0 if is_rush_hour else 17.0
    
    minutes = max(5, int(round((distance_km / speed) * 60)))
    min_range = max(4, minutes - 4)
    max_range = minutes + 6
    return f"{min_range}–{max_range} min"

def predict_single_fare(p_lon, p_lat, d_lon, d_lat, p_count, trip_dt):
    dist_km = haversine_distance(p_lon, p_lat, d_lon, d_lat)
    
    input_df = pd.DataFrame([{
        "pickup_longitude": p_lon,
        "pickup_latitude": p_lat,
        "dropoff_longitude": d_lon,
        "dropoff_latitude": d_lat,
        "passenger_count": p_count,
        "distance_km": dist_km,
        "year": trip_dt.year,
        "month": trip_dt.month,
        "day": trip_dt.day,
        "hour": trip_dt.hour,
        "day_of_week": trip_dt.weekday()
    }])[features_list]

    scaled = scaler.transform(input_df)
    raw_fare = model.predict(scaled, verbose=0)[0][0]
    final_fare = max(2.50, float(raw_fare))
    return final_fare, dist_km, input_df

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="background: #0f172a; border-radius: 10px; padding: 18px; color: white; margin-bottom: 20px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size: 24px;">🚕</span>
                <div>
                    <div style="font-size: 15px; font-weight: 700; letter-spacing: 0.5px;">METROCAB NYC</div>
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase;">Fare Intelligence System</div>
                </div>
            </div>
            <div style="font-size: 11px; color: #cbd5e1; border-top: 1px solid #334155; padding-top: 8px; margin-top: 10px;">
                TLC-calibrated fare regression engine trained on historical New York City trip data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("##### Standard TLC Fare Rules")
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #475569; line-height: 1.6;">
        • <b>Base Flag Drop:</b> $2.50 upon entry<br>
        • <b>Standard Rate:</b> ~$1.55 per km in open traffic<br>
        • <b>Peak Rush Hour:</b> +$1.00 (Mon–Fri, 4 PM – 8 PM)<br>
        • <b>Overnight Window:</b> +$0.50 (8 PM – 6 AM)<br>
        • <b>State & Improvement Surcharge:</b> $0.80
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    st.markdown("##### System Environment")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #64748b;">
        • <b>Engine:</b> Keras 4-Layer Perceptron<br>
        • <b>Normalization:</b> Fit StandardScaler<br>
        • <b>Project:</b> SDP Project 2
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# TOP BRANDING BAR
# ============================================================

st.markdown(
    """
    <div class="brand-bar">
        <div>
            <div class="brand-title">🚕 MetroCab NYC <span style="font-weight: 400; color: #64748b; font-size: 1.1rem;">| Trip Fare Calculator</span></div>
            <div class="section-caption">Accurate fare estimates for New York City taxi trips based on deep regression modeling.</div>
        </div>
        <div style="display:flex; gap: 0.5rem;">
            <span class="brand-tag">TLC NYC Standard</span>
            <span class="brand-tag">Test MAE: $1.87</span>
            <span class="brand-tag">R²: 0.780</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# NAVIGATION TABS
# ============================================================

tab_calc, tab_batch, tab_specs = st.tabs([
    "Fare Calculator",
    "Batch Evaluation",
    "Model Specifications & Benchmarks"
])

# ============================================================
# TAB 1: FARE CALCULATOR
# ============================================================

with tab_calc:
    # Initialize session state for locations and calculation results
    if "pickup_loc_val" not in st.session_state:
        st.session_state.pickup_loc_val = "Times Square (Midtown Manhattan)"
    if "dropoff_loc_val" not in st.session_state:
        st.session_state.dropoff_loc_val = "JFK International Airport (Terminal 4)"
    if "calc_timestamp" not in st.session_state:
        st.session_state.calc_timestamp = None

    def on_preset_selected():
        selected = st.session_state.get("preset_picker_key", "")
        if selected in PRESET_ROUTES:
            p_val, d_val = PRESET_ROUTES[selected]
            st.session_state.pickup_loc_val = p_val
            st.session_state.dropoff_loc_val = d_val

    # Route Preset selector
    st.markdown("##### Quick Route Selection")
    preset_choice = st.selectbox(
        "Select a frequent NYC route or customize below:",
        ["Select a route preset..."] + list(PRESET_ROUTES.keys()),
        key="preset_picker_key",
        on_change=on_preset_selected
    )

    col_route, col_time = st.columns([1.1, 0.9], gap="large")
    
    with col_route:
        st.markdown("##### Route Origin & Destination")
        
        pickup_loc = st.selectbox(
            "Pickup Location",
            options=list(NYC_LOCATIONS.keys()),
            key="pickup_loc_val"
        )

        col_swap, _ = st.columns([1.2, 2])
        with col_swap:
            if st.button("⇄ Swap Direction", help="Swap pickup and destination", use_container_width=True, key="btn_swap_direction"):
                p_curr = st.session_state.pickup_loc_val
                d_curr = st.session_state.dropoff_loc_val
                st.session_state.pickup_loc_val = d_curr
                st.session_state.dropoff_loc_val = p_curr
                st.session_state.pop("cached_fare_val", None)
                st.rerun()
        
        dropoff_loc = st.selectbox(
            "Drop-off Destination",
            options=list(NYC_LOCATIONS.keys()),
            key="dropoff_loc_val"
        )
        
        # Airport corridor detection
        is_jfk = ("JFK" in pickup_loc) or ("JFK" in dropoff_loc)
        is_lga = ("LaGuardia" in pickup_loc) or ("LaGuardia" in dropoff_loc)
        is_manhattan_jfk = is_jfk and any(m in (pickup_loc + dropoff_loc) for m in ["Midtown", "Central Park", "Wall Street", "Empire State", "Times Square", "SoHo", "Grand Central", "Penn Station"])

        if is_manhattan_jfk:
            st.markdown('<div class="condition-pill condition-airport">✈️ <b>TLC Airport Corridor:</b> Manhattan ↔ JFK trips operate under regulated flat-rate pricing ($52.00 base historically, $70.00 base currently + surcharges). The neural network reflects this corridor rate.</div>', unsafe_allow_html=True)
        elif is_lga:
            st.markdown('<div class="condition-pill condition-airport">✈️ <b>TLC Airport Corridor:</b> LaGuardia Airport (LGA) trips follow standard mileage plus bridge/tunnel tolls.</div>', unsafe_allow_html=True)
        
        sel_p_lat, sel_p_lon = NYC_LOCATIONS[pickup_loc]
        sel_d_lat, sel_d_lon = NYC_LOCATIONS[dropoff_loc]

        use_custom = st.checkbox("Manual GPS coordinates override", value=False)
        if use_custom:
            c_p1, c_p2 = st.columns(2)
            p_lat = c_p1.number_input("Pickup Latitude", value=float(sel_p_lat), format="%.6f", key="c_p_lat")
            p_lon = c_p2.number_input("Pickup Longitude", value=float(sel_p_lon), format="%.6f", key="c_p_lon")
            d_lat = c_p1.number_input("Drop-off Latitude", value=float(sel_d_lat), format="%.6f", key="c_d_lat")
            d_lon = c_p2.number_input("Drop-off Longitude", value=float(sel_d_lon), format="%.6f", key="c_d_lon")
        else:
            p_lat, p_lon = sel_p_lat, sel_p_lon
            d_lat, d_lon = sel_d_lat, sel_d_lon

    with col_time:
        st.markdown("##### Departure & Occupancy")
        
        col_d, col_t = st.columns(2)
        with col_d:
            ride_date = st.date_input("Date", value=datetime.date(2015, 6, 1))
        with col_t:
            ride_time = st.time_input("Departure Time", value=datetime.time(17, 30))
            
        passengers = st.slider("Number of Passengers", min_value=1, max_value=6, value=1)
        
        trip_datetime = datetime.datetime.combine(ride_date, ride_time)
        
        is_rush = (16 <= trip_datetime.hour <= 19) and (trip_datetime.weekday() < 5)
        is_night = (trip_datetime.hour >= 20) or (trip_datetime.hour < 6)
        
        if is_rush:
            st.markdown('<div class="condition-pill condition-rush">🕒 Peak Rush Hour Window (+$1.00 TLC Surcharge)</div>', unsafe_allow_html=True)
        elif is_night:
            st.markdown('<div class="condition-pill condition-night">🌙 Overnight Window (+$0.50 TLC Surcharge)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="condition-pill condition-normal">☀️ Standard Traffic Window (Off-Peak Rates)</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    btn_calc = st.button("🚕 Calculate Estimated Fare", type="primary", use_container_width=True, key="btn_run_fare_calc")
    
    # Run calculation and store in session state
    if btn_calc or ("cached_fare_val" not in st.session_state):
        with st.spinner("Calculating fare using Deep Neural Network..."):
            fare_val, trip_dist_km, feat_vector = predict_single_fare(
                p_lon, p_lat, d_lon, d_lat, passengers, trip_datetime
            )
            st.session_state.cached_fare_val = fare_val
            st.session_state.cached_trip_dist_km = trip_dist_km
            st.session_state.cached_feat_vector = feat_vector
            st.session_state.calc_timestamp = datetime.datetime.now().strftime("%I:%M:%S %p")
            if btn_calc:
                st.toast(f"Fare calculated: ${fare_val:.2f}", icon="🚕")

    fare_val = st.session_state.cached_fare_val
    trip_dist_km = st.session_state.cached_trip_dist_km
    feat_vector = st.session_state.cached_feat_vector
    calc_time = st.session_state.calc_timestamp

    trip_dist_mi = trip_dist_km * 0.621371
    duration_str = calculate_duration(trip_dist_km, is_rush)
    
    if btn_calc:
        st.success(f"✓ Fare calculated successfully: **${fare_val:.2f}** for **{pickup_loc.split('(')[0].strip()} ➡️ {dropoff_loc.split('(')[0].strip()}** (Calculated at {calc_time})")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Results Presentation
    res_col, map_col = st.columns([1, 1.25], gap="large")
    
    with res_col:
        st.markdown("##### Fare Summary")
        
        low_est = max(2.50, fare_val - 1.87)
        high_est = fare_val + 1.87
        
        traffic_note = "Peak traffic surcharge included" if is_rush else ("Night surcharge included" if is_night else "Standard regular rate")
        
        st.markdown(
            f"""
            <div class="receipt-card">
                <div class="receipt-header">
                    <div>
                        <div class="receipt-title">Estimated Total Fare</div>
                        <div class="receipt-price">${fare_val:.2f}</div>
                    </div>
                    <div class="receipt-badge">Expected: ${low_est:.2f} – ${high_est:.2f}</div>
                </div>
                <div class="receipt-divider"></div>
                <div class="receipt-row">
                    <span>Direct Distance</span>
                    <b>{trip_dist_mi:.1f} miles ({trip_dist_km:.1f} km)</b>
                </div>
                <div class="receipt-row">
                    <span>Estimated Transit Time</span>
                    <b>{duration_str}</b>
                </div>
                <div class="receipt-row">
                    <span>Effective Rate</span>
                    <b>${fare_val / max(0.1, trip_dist_km):.2f} / km</b>
                </div>
                <div class="receipt-row">
                    <span>Occupancy</span>
                    <b>{passengers} {'passenger' if passengers == 1 else 'passengers'}</b>
                </div>
                <div class="receipt-row">
                    <span>Pricing Context</span>
                    <b>{traffic_note}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        receipt_text = f"""
============================================================
              METROCAB NYC | TRIP FARE RECEIPT
============================================================
Date / Time      : {trip_datetime.strftime('%Y-%m-%d %I:%M %p')}
Pickup Location  : {pickup_loc}
Drop-off Location: {dropoff_loc}
Passengers       : {passengers}
Direct Distance  : {trip_dist_mi:.2f} miles ({trip_dist_km:.2f} km)
Estimated Time   : {duration_str}
------------------------------------------------------------
ESTIMATED FARE   : ${fare_val:.2f}
Expected Range   : ${low_est:.2f} - ${high_est:.2f} (±$1.87 MAE)
Rate per KM      : ${fare_val / max(0.1, trip_dist_km):.2f}/km
Traffic Window   : {traffic_note}
------------------------------------------------------------
Model: 4-Layer Deep Feedforward Neural Network (SDP Project 2)
Test MAE: $1.87 | R²: 0.7804 | NYC TLC Baseline
============================================================
""".strip()

        st.download_button(
            label="📄 Download Ride Receipt (TXT)",
            data=receipt_text,
            file_name=f"metrocab_receipt_{trip_datetime.strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with map_col:
        map_h_col, theme_col = st.columns([1.2, 1.2])
        with map_h_col:
            st.markdown("##### Route Geography")
        with theme_col:
            map_theme = st.radio(
                "Map Theme",
                ["☀️ Light", "🌙 Midnight Dark"],
                horizontal=True,
                label_visibility="collapsed"
            )
        
        is_dark_map = "Midnight" in map_theme
        map_style_str = "dark" if is_dark_map else "light"
        arc_source_color = [250, 204, 21, 230] if is_dark_map else [234, 179, 8, 220]
        arc_target_color = [251, 146, 60, 230] if is_dark_map else [249, 115, 22, 220]
        
        # PyDeck mapping with proper auto-zoom framing
        mid_lat = (p_lat + d_lat) / 2.0
        mid_lon = (p_lon + d_lon) / 2.0
        
        zoom = 11.2 if trip_dist_km > 15 else (12.2 if trip_dist_km > 7 else 13.2)
        
        points_data = pd.DataFrame([
            {"lat": p_lat, "lon": p_lon, "label": f"Pickup: {pickup_loc}", "color": [16, 185, 129, 240]},
            {"lat": d_lat, "lon": d_lon, "label": f"Drop-off: {dropoff_loc}", "color": [239, 68, 68, 240]}
        ])
        
        arc_data = pd.DataFrame([{
            "src": [p_lon, p_lat],
            "dst": [d_lon, d_lat]
        }])
        
        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=points_data,
                get_position="[lon, lat]",
                get_color="color",
                get_radius=130,
                pickable=True
            ),
            pdk.Layer(
                "ArcLayer",
                data=arc_data,
                get_source_position="src",
                get_target_position="dst",
                get_source_color=arc_source_color,
                get_target_color=arc_target_color,
                get_width=4
            )
        ]
        
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=zoom,
                pitch=20
            ),
            map_style=map_style_str,
            tooltip={"text": "{label}"}
        )
        st.pydeck_chart(deck, use_container_width=True)

    with st.expander("Technical: Input Feature Vector"):
        st.dataframe(feat_vector.style.format("{:.4f}"), use_container_width=True)

# ============================================================
# TAB 2: BATCH EVALUATION
# ============================================================

with tab_batch:
    st.markdown("##### Batch Inference & Out-of-Sample Testing")
    st.markdown("Evaluate multiple trip records simultaneously or simulate against test holdouts.")
    
    col_b1, col_b2 = st.columns(2, gap="medium")
    
    with col_b1:
        st.markdown("###### Load from Test Dataset")
        n_samples = st.selectbox("Sample size from holdout test set:", [5, 10, 25, 50], index=1)
        btn_run_test = st.button("Run Simulation on Test Records")
        
    with col_b2:
        st.markdown("###### Upload Custom CSV")
        csv_file = st.file_uploader("Upload CSV containing trip coordinates and timestamps", type=["csv"])

    eval_df = None
    
    if btn_run_test:
        if TEST_DATA_PATH.exists():
            eval_df = pd.read_csv(TEST_DATA_PATH, nrows=n_samples)
        else:
            st.error("Test dataset file not found at DATA/raw/test.csv.")
            
    elif csv_file is not None:
        try:
            eval_df = pd.read_csv(csv_file)
        except Exception as ex:
            st.error(f"Error parsing uploaded file: {ex}")

    if eval_df is not None:
        try:
            dt_series = pd.to_datetime(eval_df["pickup_datetime"])
            eval_df["distance_km"] = haversine_distance(
                eval_df["pickup_longitude"], eval_df["pickup_latitude"],
                eval_df["dropoff_longitude"], eval_df["dropoff_latitude"]
            )
            eval_df["year"] = dt_series.dt.year
            eval_df["month"] = dt_series.dt.month
            eval_df["day"] = dt_series.dt.day
            eval_df["hour"] = dt_series.dt.hour
            eval_df["day_of_week"] = dt_series.dt.dayofweek
            
            X_eval = eval_df[features_list]
            X_eval_scaled = scaler.transform(X_eval)
            batch_predictions = model.predict(X_eval_scaled, verbose=0).flatten()
            eval_df["predicted_fare"] = np.round(np.clip(batch_predictions, 2.50, None), 2)
            
            # Summary Metrics
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Trips Processed", len(eval_df))
            m2.metric("Total Fare Volume", f"${eval_df['predicted_fare'].sum():,.2f}")
            m3.metric("Mean Trip Fare", f"${eval_df['predicted_fare'].mean():.2f}")
            m4.metric("Avg Distance", f"{eval_df['distance_km'].mean():.2f} km")
            
            cols_show = ["pickup_datetime", "passenger_count", "distance_km", "predicted_fare"]
            if "key" in eval_df.columns:
                cols_show = ["key"] + cols_show
            st.dataframe(eval_df[cols_show], use_container_width=True)
            
            csv_export = eval_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed CSV", data=csv_export, file_name="metro_cab_predictions.csv", mime="text/csv")
            
        except Exception as err:
            st.error(f"Processing error: {err}")

# ============================================================
# TAB 3: MODEL SPECIFICATIONS & BENCHMARKS
# ============================================================

with tab_specs:
    st.markdown("##### Model Architecture & Empirical Benchmarks")
    
    st.markdown(
        """
        The production model is a **Deep Feedforward Neural Network** with four dense layers, optimized using the Adam algorithm.
        Input features comprise 11 spatial, temporal, and occupancy variables normalized via a training-fitted StandardScaler.
        """
    )
    
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp_data = pd.read_csv(comp_path)
        st.markdown("###### Empirical Validation Performance Across Iterations")
        st.dataframe(
            comp_data.style.format({
                "MAE": "${:.4f}",
                "RMSE": "${:.4f}",
                "R2": "{:.4f}"
            }),
            use_container_width=True
        )

    st.markdown("###### Benchmark Visualizations")
    img_c1, img_c2 = st.columns(2)
    
    val_chart = RESULTS_DIR / "validation_models_comparison.png"
    dist_chart = RESULTS_DIR / "predicted_fares_distribution.png"
    
    if val_chart.exists():
        img_c1.image(str(val_chart), caption="Error & Fit Comparison: Baseline vs. Deep Neural Network Architectures")
    if dist_chart.exists():
        img_c2.image(str(dist_chart), caption="Predicted Fare Distribution over Holdout Test Population (9,914 trips)")

    st.markdown("---")
    st.markdown("###### Feature Attribution & Permutation Importance")
    st.markdown(
        """
        To explain the internal decision logic of the neural network, **Permutation Feature Importance** 
        was computed across the test population. Shuffling critical features degrades prediction accuracy, 
        revealing each variable's marginal contribution to final fare calculations:
        """
    )

    feat_c1, feat_c2 = st.columns([1.2, 1], gap="medium")
    
    feat_chart = RESULTS_DIR / "feature_importance.png"
    feat_csv = RESULTS_DIR / "feature_importance.csv"

    with feat_c1:
        if feat_chart.exists():
            st.image(str(feat_chart), caption="Permutation Feature Importance (% Contribution to Error)", use_container_width=True)

    with feat_c2:
        if feat_csv.exists():
            f_data = pd.read_csv(feat_csv)
            st.dataframe(
                f_data.style.format({
                    "MAE_Impact": "+${:.2f}",
                    "Relative_Importance_Pct": "{:.1f}%"
                }),
                use_container_width=True
            )
    
    st.markdown(
        """
        **Key Behavioral Insights:**
        - **Distance Dominance (24.8%):** Great-circle Haversine distance is the single largest isolated driver of fare amounts.
        - **Borough & Corridor Coordinates (58.2% Combined):** Longitude and latitude collectively capture location-specific density, airport surcharges, and inter-borough bridge/tunnel crossings that pure linear distance cannot explain.
        - **Macroeconomic Temporal Shifts (`year`, 6.8%):** Captures historical NYC TLC regulated base fare hikes (such as the major September 2012 rate increase).
        
        ---
        
        ###### Architectural Selection: Deep Learning vs. Tree-Based Models (GBDT)
        While Gradient Boosted Trees (XGBoost/LightGBM) are industry standards for tabular benchmarks:
        1. **Continuous Geospatial Embeddings:** Deep Feedforward Neural Networks naturally learn smooth, continuous geographical surfaces across dense coordinate meshes.
        2. **Multi-Task & Scalability:** Deep learning pipelines integrate smoothly with downstream vector search, real-time embeddings, and high-throughput streaming systems (TensorFlow Serving).
        """
    )