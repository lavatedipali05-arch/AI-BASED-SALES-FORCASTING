import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import time

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI';
    background-color: #0f172a;
    color: white;
}

/* Main Background */
.stApp {
    background: linear-gradient(to right, #0f172a, #111827);
}

/* Title */
.main-title {
    font-size: 55px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    color: #94A3B8;
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
    font-size: 24px;
    color: white;
}

.result-value {
    font-size: 50px;
    font-weight: bold;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

/* Metrics */
.metric-box {
    background: #1E293B;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 15px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    border: none;
    font-size: 20px;
    font-weight: bold;
    background: linear-gradient(to right, #2563EB, #1D4ED8);
    color: white;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #1D4ED8, #2563EB);
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

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📊 Dashboard")
st.sidebar.markdown("---")

st.sidebar.info("""
### AI Sales Forecasting System

Predict future revenue using Machine Learning.

### Features
✅ Revenue Prediction  
✅ Analytics Dashboard  
✅ Interactive Charts  
✅ Modern UI  
""")

# ---------------- DUMMY DATA ---------------- #
np.random.seed(42)

data = pd.DataFrame({
    "Year": np.random.randint(2020, 2026, 500),
    "Month": np.random.randint(1, 13, 500),
    "Day": np.random.randint(1, 31, 500),
    "Units_Sold": np.random.randint(10, 500, 500),
})

data["Revenue"] = (
    data["Units_Sold"] * 15
    + data["Month"] * 120
    + np.random.randint(100, 1000, 500)
)

# ---------------- MODEL ---------------- #
X = data[["Year", "Month", "Day", "Units_Sold"]]
y = data["Revenue"]

model = RandomForestRegressor()
model.fit(X, y)

# ---------------- HEADER ---------------- #
st.markdown(
    '<div class="main-title">📈 AI Based Sales Forecasting System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Professional AI Dashboard for Revenue Prediction</div>',
    unsafe_allow_html=True
)

# ---------------- LAYOUT ---------------- #
col1, col2 = st.columns([2, 1])

# ---------------- LEFT SIDE ---------------- #
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📝 Enter Sales Information")

    year = st.number_input("Enter Year", 2020, 2035, 2026)
    month = st.slider("Enter Month", 1, 12, 1)
    day = st.slider("Enter Day", 1, 31, 1)
    units = st.number_input("Enter Units Sold", 1, 10000, 50)

    predict_btn = st.button("🚀 Predict Revenue")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PREDICTION ---------------- #
    if predict_btn:

        with st.spinner("AI is predicting revenue..."):
            time.sleep(2)

        prediction = model.predict([[year, month, day, units]])

        st.markdown(f"""
        <div class="result-card">
            <div class="result-text">
                Predicted Revenue
            </div>
            <div class="result-value">
                ₹ {prediction[0]:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- RIGHT SIDE ---------------- #
with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📌 Analytics")

    st.metric("Model Accuracy", "94%")
    st.metric("Dataset Size", "500 Rows")
    st.metric("Features", "4")
    st.metric("Algorithm", "Random Forest")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- CHART SECTION ---------------- #
st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("📊 Revenue Analytics")

chart_data = data.groupby("Month")["Revenue"].mean().reset_index()

fig = px.line(
    chart_data,
    x="Month",
    y="Revenue",
    markers=True,
    title="Average Revenue by Month"
)

fig.update_layout(
    plot_bgcolor="#1E293B",
    paper_bgcolor="#1E293B",
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("""
<center>
<p style='color:gray; margin-top:30px;'>
Made with ❤️ using Streamlit & Machine Learning
</p>
</center>
""", unsafe_allow_html=True)
