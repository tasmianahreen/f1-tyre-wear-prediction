import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

DATA_PATH = "data/bahrain_2024_laps.csv"
MODEL_PATH = "models/tyre_degradation_model.pkl"

st.set_page_config(page_title="F1 Tyre Wear Predictor", layout="wide")

st.title("Formula 1 Tyre Wear Prediction Dashboard")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

st.sidebar.header("Prediction Inputs")

driver = st.sidebar.selectbox("Driver", sorted(df["Driver"].unique()))
team = df[df["Driver"] == driver]["Team"].iloc[0]
compound = st.sidebar.selectbox("Compound", sorted(df["Compound"].unique()))
tyre_life = st.sidebar.slider("Tyre Life", 1, 40, 10)
stint = st.sidebar.slider("Stint", 1, 5, 1)
lap_number = st.sidebar.slider("Lap Number", 1, int(df["LapNumber"].max()), 10)

input_df = pd.DataFrame(
    {
        "Driver": [driver],
        "Team": [team],
        "Compound": [compound],
        "TyreLife": [tyre_life],
        "Stint": [stint],
        "LapNumber": [lap_number],
    }
)

prediction = model.predict(input_df)[0]

st.metric("Predicted Tyre Degradation", f"{prediction:.2f} seconds")

st.subheader("Actual Tyre Degradation by Compound")

fig = px.line(
    df.groupby(["Compound", "TyreLife"], as_index=False)["Degradation"].median(),
    x="TyreLife",
    y="Degradation",
    color="Compound",
    markers=True,
    title="Median Tyre Degradation Over Tyre Life",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Driver Stint Data")

driver_df = df[df["Driver"] == driver]

fig2 = px.scatter(
    driver_df,
    x="TyreLife",
    y="LapTimeSeconds",
    color="Compound",
    hover_data=["LapNumber", "Stint"],
    title=f"{driver} Lap Times by Tyre Life",
)

st.plotly_chart(fig2, use_container_width=True)

st.dataframe(driver_df)
