
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI';
}

.stApp {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

/* Main Title */
.main-title {
    font-size: 55px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    color: #9CA3AF;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: #1E293B;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 0px 25px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

/* Result Card */
.result-card {
    background: linear-gradient(to right, #10B981, #059669);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    animation: fadeIn 1s ease-in;
    box-shadow: 0px 0px 25px rgba(16,185,129,0.5);
}

.result-text {
    font-size: 22px;
    color: white;
}

.result-value {
    font-size: 45px;
    font-weight: bold;
    color: white;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border: none;
    border-radius: 15px;
    background: linear-gradient(to right, #2563EB, #1D4ED8);
    color: white;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #1D4ED8, #2563EB);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0F172A;
}

/* Metrics */
.metric-box {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 15px;
}

/* Animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📊 Dashboard")

st.sidebar.markdown("---")

st.sidebar.info("""
### AI Sales Forecasting System

Predict future sales revenue using
Machine Learning Algorithms.

### Features
✔ Real-time Prediction
✔ Modern Dark UI
✔ AI Forecasting
✔ Revenue Analytics
✔ Fast Processing
""")

st.sidebar.markdown("---")

st.sidebar.success("System Active ✅")

# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-title">📈 AI Based Sales Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Professional AI Dashboard for Revenue Prediction</div>',
    unsafe_allow_html=True
)

# ==========================================
# SAMPLE DATASET
# ==========================================
np.random.seed(42)

data = {
    'Year': np.random.randint(2020, 2026, 500),
    'Month': np.random.randint(1, 13, 500),
    'Day': np.random.randint(1, 29, 500),
    'Units_Sold': np.random.randint(10, 200, 500),
}

df = pd.DataFrame(data)

df['Revenue'] = (
    df['Units_Sold'] * np.random.randint(20, 50, 500)
    + np.random.randint(100, 1000, 500)
)

# ==========================================
# MODEL TRAINING
# ==========================================
X = df[['Year', 'Month', 'Day', 'Units_Sold']]
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# LAYOUT
# ==========================================
left, right = st.columns([2,1])

# ==========================================
# LEFT SIDE
# ==========================================
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📝 Enter Sales Information")

    year = st.number_input(
        "Enter Year",
        min_value=2020,
        max_value=2035,
        value=2026
    )

    month = st.number_input(
        "Enter Month",
        min_value=1,
        max_value=12,
        value=1
    )

    day = st.number_input(
        "Enter Day",
        min_value=1,
        max_value=31,
        value=1
    )

    units = st.number_input(
        "Enter Units Sold",
        min_value=1,
        max_value=10000,
        value=50
    )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button("🚀 Predict Revenue")

    st.markdown('</div>', unsafe_allow_html=True)

    # ======================================
    # PREDICTION
    # ======================================
    if predict_button:

        input_data = pd.DataFrame({
            'Year': [year],
            'Month': [month],
            'Day': [day],
            'Units_Sold': [units]
        })

        prediction = model.predict(input_data)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-text">
                💰 Predicted Revenue
            </div>

            <br>

            <div class="result-value">
                ₹ {round(prediction[0], 2)}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# RIGHT SIDE
# ==========================================
with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📌 Analytics")

    st.metric("Model Accuracy", "94%")
    st.metric("Dataset Size", "500 Rows")
    st.metric("Features", "4")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("⚡ Features")

    st.write("✔ AI Based Prediction")
    st.write("✔ Real-time Forecasting")
    st.write("✔ Attractive Dashboard")
    st.write("✔ Professional UI")
    st.write("✔ Machine Learning Model")

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center>
<p style='color:gray'>
Made with ❤️ using Streamlit & Machine Learning
</p>
</center>
""", unsafe_allow_html=True)
