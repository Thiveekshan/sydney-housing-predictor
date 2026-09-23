"""
Sydney Housing Price Prediction App
------------------------------------
A Streamlit application that loads the trained Decision Tree Regressor model
and predicts a property's sale price based on user-entered features.

To run this app:
    1. Make sure 'sydney_housing_model.pkl' is in the same folder as this file.
    2. Also keep the '.streamlit' folder (containing config.toml) alongside
       this file - it controls the app's colour theme.
    3. Open a terminal (Anaconda Prompt) in that folder.
    4. Run: streamlit run app.py
    5. A browser tab will open automatically showing the app.
"""

import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# ----------------------------------------------------------------------
# Page configuration - sets the browser tab title and icon
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ----------------------------------------------------------------------
# Custom styling
# Streamlit's config.toml (in .streamlit/) sets the overall colour theme.
# Here I add extra CSS for the specific elements I build myself below
# (the hero band, input cards, and result card), imported fonts, and to
# quieten a few default Streamlit chrome elements for a cleaner look.
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        background: #142C3D;
        padding: 2.4rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
    }
    .hero h1 {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        color: #F6F5F2;
        font-size: 2.1rem;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.01em;
    }
    .hero p {
        color: #B7C2C9;
        font-size: 0.98rem;
        margin: 0;
        line-height: 1.5;
        max-width: 40rem;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #E3E0D8;
        border-radius: 12px;
        padding: 1.6rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.4rem;
    }
    .section-label {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        color: #142C3D;
        font-size: 1.15rem;
        margin-bottom: 0.9rem;
    }

    .result-card {
        background: linear-gradient(135deg, #142C3D 0%, #1D3F55 100%);
        border-radius: 14px;
        padding: 2rem 2rem 1.6rem 2rem;
        margin-top: 0.4rem;
    }
    .result-label {
        color: #B7C2C9;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    .result-price {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        color: #F6F5F2;
        font-size: 2.8rem;
        margin: 0 0 0.9rem 0;
    }
    .result-note {
        color: #93A5AE;
        font-size: 0.85rem;
        line-height: 1.5;
        border-top: 1px solid rgba(255,255,255,0.12);
        padding-top: 0.9rem;
        margin-top: 0.4rem;
    }

    .suburb-chip {
        display: inline-block;
        background: #FFFFFF;
        border: 1px solid #E3E0D8;
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        margin-right: 0.5rem;
        font-size: 0.82rem;
        color: #5B6B75;
    }
    .suburb-chip b { color: #142C3D; }

    div.stButton > button {
        background: #A6773D;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #8F6530;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load the trained model
# joblib.load() reads back the exact model object saved in the notebook
# (Section 8.1), including its learned splits and thresholds.
# ----------------------------------------------------------------------
model = joblib.load("sydney_housing_model.pkl")

# Reference context from the training dataset (notebook Section 3.2 / 4.3),
# shown to the user as helpful context alongside their prediction.
SUBURB_CONTEXT = {
    "Campbelltown": {"median": 950_500, "type": "House-dominant, affordable"},
    "Parramatta": {"median": 620_000, "type": "Unit-dominant, major hub"},
    "Randwick": {"median": 1_650_000, "type": "Mixed, premium"},
}

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Sydney Housing Price Predictor</h1>
        <p>A decision-support tool estimating sale prices across Randwick,
        Parramatta, and Campbelltown, trained on 101 sold-property records
        collected from realestate.com.au and domain.com.au.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Input section
# ----------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Property Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    suburb = st.selectbox("Suburb", ["Campbelltown", "Parramatta", "Randwick"])
    property_type = st.selectbox("Property Type", ["House", "Townhouse", "Unit"])
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=12, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2, step=1)
with col2:
    car_spaces = st.number_input("Car Spaces", min_value=0, max_value=6, value=1, step=1)
    size_sqm = st.number_input("Size (sqm)", min_value=20.0, max_value=1500.0, value=150.0, step=5.0)
    distance_to_cbd_km = st.number_input(
        "Distance to Sydney CBD (km)", min_value=0.0, max_value=100.0, value=25.0, step=0.5
    )

# Suburb context chips - quick reference so the user can sanity-check
# their prediction against typical suburb pricing before they even predict.
ctx = SUBURB_CONTEXT[suburb]
st.markdown(
    f"""
    <div style="margin: 0.6rem 0 1.2rem 0;">
        <span class="suburb-chip">Suburb median: <b>${ctx['median']:,.0f}</b></span>
        <span class="suburb-chip">{ctx['type']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

predict_clicked = st.button("Predict Sale Price")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if predict_clicked:

    # sale_month: uses the current month, reflecting "today's" market timing,
    # consistent with how this feature was engineered in the notebook
    # (Section 4.2).
    sale_month = datetime.now().month

    # Convert the suburb and property_type selections into the same one-hot
    # encoded columns the model was trained on (notebook Section 4.4). The
    # model only understands these numeric 0/1 columns, not suburb names.
    suburb_parramatta = 1 if suburb == "Parramatta" else 0
    suburb_randwick = 1 if suburb == "Randwick" else 0
    property_type_townhouse = 1 if property_type == "Townhouse" else 0
    property_type_unit = 1 if property_type == "Unit" else 0

    # Feature order must exactly match training (notebook Section 8.1):
    # ['bedrooms', 'bathrooms', 'car_spaces', 'size_sqm', 'distance_to_cbd_km',
    #  'sale_month', 'suburb_Parramatta', 'suburb_Randwick',
    #  'property_type_Townhouse', 'property_type_Unit']
    features = np.array([[
        bedrooms, bathrooms, car_spaces, size_sqm, distance_to_cbd_km,
        sale_month, suburb_parramatta, suburb_randwick,
        property_type_townhouse, property_type_unit,
    ]])

    predicted_price = model.predict(features)[0]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated Sale Price</div>
            <div class="result-price">${predicted_price:,.0f}</div>
            <div class="result-note">
                Based on a Decision Tree Regressor trained on 101 properties
                sold in Randwick, Parramatta, and Campbelltown. Predictions
                for properties unlike those in the training data — unusually
                large, premium, or renovated properties — should be treated
                with caution. See the accompanying report for a full
                discussion of model limitations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "This tool only supports Randwick, Parramatta, and Campbelltown, as "
    "these are the only suburbs the underlying model was trained on."
)
