from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import os
import pandas as pd
import numpy as np
import cv2
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from supabase import create_client, Client

app = FastAPI()

# Environment-based Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CITY = "Chengalpattu"
COUNTRY = "Tamil Nadu"
THRESHOLD_MARGIN = 5.0
TREND_DAYS = 3
TEMP_MIN = 10.0
TEMP_MAX = 50.0

WEATHER_CONDITIONS = {
    "Rain": "Don't water, rain expected",
    "Heavy Rain": "No watering needed, heavy rain expected",
    "Sunny": "Mild risk, monitor temperature",
    "Clear": "Consider watering if temperature is high",
    "Extreme": "High risk, water immediately"
}

# Utility Functions

def extract_temperature_from_thermal_video(video_path):
    cap = cv2.VideoCapture(video_path)
    temps = []
    frame_count = 0
    if not cap.isOpened():
        return None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_intensity = np.mean(gray)
            temp = TEMP_MIN + ((avg_intensity / 255.0) * (TEMP_MAX - TEMP_MIN))
            temps.append(temp)
    cap.release()
    result = sum(temps) / len(temps) if temps else None
    print(f"Extracted average temperature from video: {result}")
    return result

def load_history():
    response = supabase.table("temperature_history").select("*").execute()
    data = response.data
    if data:
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded history from Supabase: {len(df)} records")
        return df
    print("No history in Supabase, returning empty DataFrame")
    return pd.DataFrame(columns=['date', 'temperature', 'watered'])

def compute_baseline(history):
    if not history.empty:
        watered_temps = history[history['watered'] == True]['temperature']
        baseline = watered_temps.mean() if not watered_temps.empty else 28.0
        print(f"Computed baseline from history: {baseline}")
        return baseline
    print("History is empty, default baseline: 28.0")
    return 28.0

def compute_trend(history):
    if len(history) < TREND_DAYS:
        print("Not enough history for trend computation")
        return 0.0
    df = history.sort_values('date').tail(30)
    temps = df['temperature'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    temps_scaled = scaler.fit_transform(temps)
    X, y = [], []
    for i in range(5, len(temps_scaled)):
        X.append(temps_scaled[i-5:i])
        y.append(temps_scaled[i])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        print("Not enough data to build trend model")
        return 0.0
    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)
    prediction = model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]), verbose=0)
    predicted_temp = scaler.inverse_transform(prediction)[0][0]
    last_temp = df['temperature'].values[-1]
    trend = predicted_temp - last_temp
    print(f"Predicted temp: {predicted_temp}, Last temp: {last_temp}, Trend: {trend}")
    return trend

def train_anomaly_detector(history):
    if len(history[history['watered'] == True]) < 5:
        print("Not enough data to train anomaly detector")
        return None
    X = history[history['watered'] == True]['temperature'].values.reshape(-1, 1)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    print("Trained anomaly detector")
    return model

def is_anomalous_temperature(current_temp, model):
    if model is None:
        return False
    prediction = model.predict(np.array([[current_temp]]))[0] == -1
    print(f"Anomaly detection result for {current_temp}: {prediction}")
    return prediction

def get_weather_forecast():
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY},{COUNTRY}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != "200":
            return None, None
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        temps, conditions = [], []
        for entry in data.get("list", []):
            if "main" in entry and "weather" in entry and entry.get("dt_txt"):
                forecast_date = datetime.strptime(entry["dt_txt"], "%Y-%m-%d %H:%M:%S").date()
                if forecast_date == tomorrow:
                    temps.append(entry["main"].get("temp", 0))
                    if entry["weather"]:
                        conditions.append(entry["weather"][0].get("main", "Unknown"))
        max_temp = max(temps) if temps else None
        common_condition = max(set(conditions), key=conditions.count) if conditions else "Unknown"
        print(f"Forecast for tomorrow: Temp={max_temp}, Condition={common_condition}")
        return max_temp, common_condition
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None, None

def check_paddy_temp_condition(current_temp):
    print("Skipping paddy temp reference check (could be added via Supabase Storage)")
    return False

def is_thermal_temp_unreliable(current_temp, forecast_temp, margin=7.0):
    unreliable = forecast_temp is not None and abs(current_temp - forecast_temp) > margin
    print(f"Thermal unreliability: {unreliable}")
    return unreliable

def needs_water(current_temp, watered, baseline, anomaly_model, trend_slope, forecast_temp, weather_condition):
    conditions_triggered = []
    print(f"Evaluating need for water: Current Temp = {current_temp}, Watered = {watered}, Baseline = {baseline}")
    if watered:
        return False, "Already watered today.", conditions_triggered
    if current_temp > (baseline + THRESHOLD_MARGIN):
        conditions_triggered.append("Baseline threshold crossed")
    if is_anomalous_temperature(current_temp, anomaly_model):
        conditions_triggered.append("Anomaly detected")
    if trend_slope > 0.1:
        conditions_triggered.append("Upward trend")
    if forecast_temp and forecast_temp > (baseline + THRESHOLD_MARGIN):
        conditions_triggered.append("Forecast high temp")
    if check_paddy_temp_condition(current_temp):
        conditions_triggered.append("Thermal condition")
    weather_advice = WEATHER_CONDITIONS.get(weather_condition, "No specific advice")
    print(f"Conditions triggered: {conditions_triggered}")
    if len(conditions_triggered) >= 2:
        return True, f"High temperature detected. {weather_advice}", conditions_triggered
    return False, weather_advice, conditions_triggered

def log_alert(current_temp, reason, conditions_triggered, force=False):
    now = datetime.now()
    data = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "temperature": current_temp,
        "reason": reason,
        "conditions_triggered": ", ".join(conditions_triggered) if conditions_triggered else "None"
    }
    supabase.table("alert_log").insert(data).execute()
    print(f"Logged alert to Supabase: {data}")

def last_log_within_24hrs():
    try:
        response = supabase.table("alert_log").select("*").order("id", desc=True).limit(1).execute()
        if response.data:
            row = response.data[0]
            last_entry = datetime.strptime(f"{row['date']} {row['time']}", "%Y-%m-%d %H:%M:%S")
            within_24hrs = datetime.now() - last_entry < timedelta(hours=24)
            print(f"Last log within 24hrs (Supabase): {within_24hrs}")
            return within_24hrs
    except Exception as e:
        print(f"Error reading last log: {e}")
    return False

def calculate_watering_frequency():
    response = supabase.table("alert_log").select("*").execute()
    data = response.data
    if not data or len(data) < 2:
        return "Not enough data to determine frequency."
    df = pd.DataFrame(data)
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        intervals = df['date'].diff().dt.days.dropna()
        if intervals.empty:
            return "Not enough valid intervals to determine frequency."
        avg_interval = round(intervals.mean())
        next_date = df['date'].max() + timedelta(days=avg_interval)
        forecast_temp, _ = get_weather_forecast()
        if forecast_temp and forecast_temp < 25:
            next_date += timedelta(days=1)
        print(f"Avg interval (Supabase): {avg_interval} days. Next: {next_date}")
        return f"Water every {avg_interval} days. Next watering: {next_date.strftime('%Y-%m-%d')}"
    except Exception as e:
        print(f"Error calculating watering frequency: {e}")
        return "Error in frequency calculation."

@app.post("/analyze")
async def analyze_field(video: UploadFile = File(...), watered: bool = Form(...)):
    with open("uploaded_video.mp4", "wb") as f:
        f.write(await video.read())
    current_temp = extract_temperature_from_thermal_video("uploaded_video.mp4")
    history = load_history()
    baseline = compute_baseline(history)
    trend_slope = compute_trend(history)
    anomaly_model = train_anomaly_detector(history)
    forecast_temp, weather_condition = get_weather_forecast()
    thermal_warning = is_thermal_temp_unreliable(current_temp, forecast_temp)
    alert, reason, conditions = needs_water(current_temp, watered, baseline, anomaly_model, trend_slope, forecast_temp, weather_condition)
    if alert:
        log_alert(current_temp, reason, conditions)
    elif not last_log_within_24hrs():
        log_alert(current_temp, "Daily log - No action required", [], force=True)
    frequency = calculate_watering_frequency()
    print(f"Analysis complete: Temp={current_temp}, Needs Water={alert}, Reason={reason}")
    return JSONResponse({
        "temperature": round(current_temp, 2),
        "needs_water": alert,
        "reason": reason,
        "conditions_triggered": conditions,
        "thermal_warning": thermal_warning,
        "weather_condition": weather_condition,
        "forecast_temperature": forecast_temp,
        "watering_frequency": frequency
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
