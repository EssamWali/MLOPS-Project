import dash
import sys
from dash import dcc  # 🟢 FIX
from dash import html  # 🟢 FIX
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from pathlib import Path
import os
import random
import joblib

# Suppress warnings about Dash's use of typing_extensions during startup
import warnings

warnings.filterwarnings("ignore")

# --- FILE PATH SETUP ---
# Define the base directory for model files
MODELS_DIR = str(Path(__file__).parent.parent.parent / "models")
DATA_DIR = str(Path(__file__).parent.parent.parent / "data" / "raw")

# --- MODEL IMPORTS ---
# Dynamically add the parent directory to the path to find model files
sys.path.append(str(Path(__file__).parent.parent))

from models.wearable_model import WearableHealthRiskModel
from models.air_quality_model import AirQualityHealthRiskModel
from models.weather_model import WeatherHealthRiskModel

# --- DATA LOADING ---
# Load all raw data once for visualization and simulation
try:
    df_wearable = pd.read_csv(
        os.path.join(DATA_DIR, "wearable_data.csv"), parse_dates=["timestamp"]
    )
    df_air = pd.read_csv(
        os.path.join(DATA_DIR, "air_quality_data.csv"), parse_dates=["timestamp"]
    )
    df_weather = pd.read_csv(
        os.path.join(DATA_DIR, "weather_data.csv"), parse_dates=["timestamp"]
    )

    # Merge data for full-system analysis (simplified merge on timestamp/location)
    df_combined = df_wearable.merge(df_air, on="timestamp", how="left").merge(
        df_weather, on="timestamp", how="left"
    )
    df_combined = df_combined.dropna(
        subset=["heart_rate", "aqi"]
    )  # Drop rows if core data is missing

    # Calculate unique metrics
    TOTAL_USERS = df_wearable["user_id"].nunique()
    TOTAL_CITIES = df_air["city"].nunique()
except FileNotFoundError as e:
    print(f"Error: Data file not found. Ensure you ran collect_data.py. Details: {e}")
    df_combined = pd.DataFrame()
    TOTAL_USERS = 0
    TOTAL_CITIES = 0


# --- MODEL INITIALIZATION AND LOADING (CRITICAL FIX) ---
# Initialize models with best types (based on MODEL_TRAINING_SUMMARY.md)

try:
    # 1. Wearable Model (Assuming Gradient Boosting was best)
    wearable_model = WearableHealthRiskModel(model_type="gradient_boosting")
    # 🟢 FIX 1: Pass the directory argument
    wearable_model.load(MODELS_DIR)

    # 2. Air Quality Model (Assuming Random Forest was best)
    air_quality_model = AirQualityHealthRiskModel(model_type="random_forest")
    # 🟢 FIX 1: Pass the directory argument
    air_quality_model.load(MODELS_DIR)

    # 3. Weather Model (Assuming Random Forest was best)
    weather_model = WeatherHealthRiskModel(model_type="random_forest")
    # 🟢 FIX 1: Pass the directory argument
    weather_model.load(MODELS_DIR)

    MODELS_LOADED = True
    print(f"✅ Models loaded successfully from {MODELS_DIR}")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODELS_LOADED = False

# --- SIMULATED PREDICTION (for dashboard display) ---
# Use the best models to generate a system-level risk score on a small sample


def generate_risk_scores(df: pd.DataFrame):
    if not MODELS_LOADED or df.empty:
        df["system_risk_score"] = 0
        df["wearable_prediction"] = "N/A"
        return df

    # Predict health status using the loaded models
    # This simulates the MultiModal Fusion Model's job by averaging probabilities

    # 1. Get predictions from each model
    w_pred, _ = wearable_model.predict(df)
    air_pred, _ = air_quality_model.predict(df)
    weather_pred, _ = weather_model.predict(df)

    df["wearable_prediction"] = w_pred
    df["air_prediction"] = air_pred
    df["weather_prediction"] = weather_pred

    # Simple risk mapping (Simulating Multi-Modal Fusion)
    risk_map = {
        "normal": 1,
        "at_risk": 2,
        "ill": 3,
        "good": 1,
        "moderate": 2,
        "unhealthy": 3,
        "very_unhealthy": 4,
        "low": 1,
        "high": 3,
    }

    df["w_score"] = df["wearable_prediction"].map(lambda x: risk_map.get(x, 0))
    df["a_score"] = df["air_prediction"].map(lambda x: risk_map.get(x, 0))
    df["t_score"] = df["weather_prediction"].map(lambda x: risk_map.get(x, 0))

    df["system_risk_score"] = (
        df[["w_score", "a_score", "t_score"]].mean(axis=1).round(1)
    )

    return df


df_full_analysis = generate_risk_scores(df_combined.copy())

# --- DASHBOARD LAYOUT & COMPONENTS ---
app = dash.Dash(__name__)

server = app.server


# Define Kpi Cards
def serve_kpi_card(title, value, color="info"):
    return html.Div(
        className="four columns",
        children=[
            html.Div(
                className=f"kpi-card {color}",
                children=[
                    html.H6(title, className="kpi-title"),
                    html.P(value, className="kpi-value"),
                ],
            )
        ],
    )


# Dashboard Features
# (Remains largely the same, focusing on the core visuals)

# --- LAYOUT (Health Authority Tab) ---
health_authority_layout = html.Div(
    [
        html.Div(
            className="row",
            children=[
                serve_kpi_card("Total Users Monitored", f"{TOTAL_USERS}", "primary"),
                serve_kpi_card(
                    "Cities/Locations Monitored", f"{TOTAL_CITIES}", "success"
                ),
                serve_kpi_card(
                    "Average System Risk (1-4)",
                    f"{df_full_analysis['system_risk_score'].mean():.2f}",
                    "warning",
                ),
                serve_kpi_card(
                    "Max Risk Alerts",
                    f"{len(df_full_analysis[df_full_analysis['system_risk_score'] >= 3])}",
                    "danger",
                ),
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="six columns chart-container",
                    children=[
                        dcc.Graph(
                            id="risk-distribution-chart",
                            figure=px.histogram(
                                df_full_analysis,
                                x="system_risk_score",
                                title="System Risk Score Distribution",
                                height=400,
                                color_discrete_sequence=["#FF7F0E"],
                            ),
                        )
                    ],
                ),
                html.Div(
                    className="six columns chart-container",
                    children=[
                        dcc.Graph(
                            id="air-quality-map",
                            figure=px.scatter_mapbox(
                                df_air.sample(min(20, len(df_air))),
                                lat="latitude",
                                lon="longitude",
                                color="aqi",
                                size="aqi",
                                mapbox_style="carto-positron",
                                zoom=3,
                                title="Air Quality Index (AQI) by Location",
                                color_continuous_scale=px.colors.sequential.Inferno,
                            ),
                        )
                    ],
                ),
            ],
        ),
        # Time Series Plot
        html.Div(
            className="row chart-container",
            children=[
                dcc.Graph(
                    id="time-series-trends",
                    figure=px.line(
                        df_full_analysis.set_index("timestamp")
                        .resample("D")
                        .mean()
                        .reset_index(),
                        x="timestamp",
                        y=["heart_rate", "stress_level"],
                        title="Daily Averages: Heart Rate & Stress",
                        height=400,
                    ),
                )
            ],
        ),
    ]
)


# --- LAYOUT (Citizen Tab) ---
citizen_layout = html.Div(
    [
        html.H4(
            "Individual Citizen Health Overview (Simulated)", className="section-title"
        ),
        # User Selection Dropdown
        dcc.Dropdown(
            id="user-selector",
            options=[
                {"label": f"User ID: {i}", "value": i}
                for i in df_wearable["user_id"].unique()
            ],
            value=df_wearable["user_id"].iloc[0] if not df_wearable.empty else None,
            style={"margin-bottom": "20px"},
        ),
        # KPI Cards for Selected User
        html.Div(id="citizen-kpi-row", className="row"),
        # Individual Metrics Chart
        html.Div(
            className="row chart-container",
            children=[dcc.Graph(id="individual-metrics-chart")],
        ),
    ]
)


app.layout = html.Div(
    children=[
        html.H1(
            "Health Risk Prediction MLOps Dashboard", style={"textAlign": "center"}
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-authority",
            children=[
                dcc.Tab(label="Health Authority Overview", value="tab-authority"),
                dcc.Tab(label="Citizen Personal View", value="tab-citizen"),
            ],
        ),
        html.Div(id="tabs-content", className="main-content"),
    ],
    style={"max-width": "1200px", "margin": "0 auto", "padding": "20px"},
)


# --- CALLBACKS ---


# 1. Tab Content Callback
@app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
def render_content(tab):
    if tab == "tab-authority":
        return health_authority_layout
    elif tab == "tab-citizen":
        return citizen_layout
    return html.Div("Select a tab.")


# 2. Citizen View Data Callback
@app.callback(
    [
        Output("citizen-kpi-row", "children"),
        Output("individual-metrics-chart", "figure"),
    ],
    [Input("user-selector", "value")],
)
def update_citizen_view(selected_user_id):
    if selected_user_id is None:
        return [html.P("Select a User ID.")], {}

    # Filter combined data for the selected user
    df_user = df_full_analysis[df_full_analysis["user_id"] == selected_user_id].copy()

    if df_user.empty:
        return [html.P(f"No data found for User ID {selected_user_id}.")], {}

    # Calculate KPIs
    latest_score = df_user["system_risk_score"].iloc[-1]
    latest_risk_level = (
        "High" if latest_score >= 3 else ("Moderate" if latest_score >= 2 else "Low")
    )
    latest_hr = df_user["heart_rate"].iloc[-1]
    latest_stress = df_user["stress_level"].iloc[-1]

    kpi_children = [
        serve_kpi_card(
            "Current Risk Score",
            f"{latest_score:.1f}",
            "danger" if latest_score >= 3 else "warning",
        ),
        serve_kpi_card(
            "Overall Risk Level",
            latest_risk_level,
            "danger" if latest_score >= 3 else "warning",
        ),
        serve_kpi_card("Current Heart Rate", f"{latest_hr:.0f} BPM", "primary"),
        serve_kpi_card("Current Stress Level", f"{latest_stress:.1f}", "info"),
    ]

    # Create detailed time-series chart
    chart_figure = px.line(
        df_user,
        x="timestamp",
        y=["heart_rate", "steps", "sleep_hours", "stress_level", "system_risk_score"],
        title=f"User {selected_user_id} - Daily Health Metrics and Predicted Risk",
        height=500,
    )

    return kpi_children, chart_figure


# --- RUN SERVER (FINAL FIX) ---
if __name__ == "__main__":
    # 🟢 FIX 2: Use the modern app.run() function
    app.run(debug=True, port=8050, host="0.0.0.0")
