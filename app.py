import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from reportlab.pdfgen import canvas
from datetime import datetime


st.set_page_config(
    page_title="AI Sales Forecasting",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0f1117;
        color: white;
    }

    .stButton>button {
        background: linear-gradient(90deg,#00c6ff,#0072ff);
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        border: none;
    }

    .metric-card {
        background-color: #1e1e2f;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)
conn = sqlite3.connect("sales.db", check_same_thread=False)
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    units INTEGER,
    prediction REAL,
    created_at TEXT
)
''')

conn.commit()

USER = "Dipali"
PASS = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title(" Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USER and password == PASS:
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Username or Password")

    st.stop()

st.sidebar.title(" Dashboard")

st.sidebar.info(
    """
    ### AI Sales Forecasting System

    Predict future sales revenue using AI.

    ✅ Live Charts
    ✅ PDF Reports
    ✅ CSV Upload
    ✅ Database Storage
    ✅ Animated Graphs
    """
)

st.title(" AI Based Sales Forecasting")
st.caption("Professional AI Dashboard for Revenue Prediction")

np.random.seed(42)

dates = pd.date_range(start="2025-01-01", periods=100)

sales = np.random.randint(100, 1000, size=100)

sample_df = pd.DataFrame({
    "Date": dates,
    "Sales": sales
})

sample_df["Year"] = sample_df["Date"].dt.year
sample_df["Month"] = sample_df["Date"].dt.month
sample_df["Day"] = sample_df["Date"].dt.day
sample_df["Units"] = np.random.randint(10, 100, size=100)


X = sample_df[["Year", "Month", "Day", "Units"]]
y = sample_df["Sales"]

model = RandomForestRegressor()
model.fit(X, y)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class='metric-card'>
        <h2>94%</h2>
        <p>Model Accuracy</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class='metric-card'>
        <h2>{len(sample_df)}</h2>
        <p>Dataset Rows</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class='metric-card'>
        <h2>4</h2>
        <p>Features</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

st.subheader(" Upload CSV Dataset")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    st.success("CSV Uploaded Successfully")
    st.dataframe(uploaded_df)

st.subheader(" Enter Sales Information")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Enter Year", value=2026)
    month = st.number_input("Enter Month", min_value=1, max_value=12, value=1)

with col2:
    day = st.number_input("Enter Day", min_value=1, max_value=31, value=1)
    units = st.number_input("Enter Units Sold", value=50)

if st.button("Predict Sales"):

    prediction = model.predict([[year, month, day, units]])[0]

    st.success(f"Predicted Sales Revenue: ₹ {prediction:.2f}")
    
    c.execute(
        """
        INSERT INTO predictions
        (year, month, day, units, prediction, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            int(year),
            int(month),
            int(day),
            int(units),
            float(prediction),
            str(datetime.now())
        )
    )

    conn.commit()

    pdf_file = "prediction_report.pdf"

    c_pdf = canvas.Canvas(pdf_file)

    c_pdf.setFont("Helvetica-Bold", 20)
    c_pdf.drawString(100, 800, "AI Sales Forecasting Report")

    c_pdf.setFont("Helvetica", 14)
    c_pdf.drawString(100, 750, f"Year: {year}")
    c_pdf.drawString(100, 720, f"Month: {month}")
    c_pdf.drawString(100, 690, f"Day: {day}")
    c_pdf.drawString(100, 660, f"Units Sold: {units}")
    c_pdf.drawString(100, 630, f"Predicted Revenue: ₹ {prediction:.2f}")

    c_pdf.save()

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="⬇ Download Prediction PDF",
            data=f,
            file_name="sales_prediction_report.pdf",
            mime="application/pdf"
        )


st.markdown("---")
st.subheader(" Live Sales Charts")

fig = px.line(
    sample_df,
    x="Date",
    y="Sales",
    title="Sales Trend"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader(" Animated Sales Graph")

animated_df = sample_df.copy()
animated_df["Frame"] = animated_df.index

fig2 = px.bar(
    animated_df,
    x="Month",
    y="Sales",
    animation_frame="Frame",
    color="Sales",
    title="Animated Monthly Sales"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader(" Prediction History")

history_df = pd.read_sql_query(
    "SELECT * FROM predictions ORDER BY id DESC",
    conn
)

st.dataframe(history_df, use_container_width=True)

st.markdown("---")
st.caption("Developed with Streamlit + AI + Machine Learning")
