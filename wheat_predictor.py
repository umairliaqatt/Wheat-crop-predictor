import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import pyttsx3

engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # female voice
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def launch_gui(model, stats):

    def greet_user():
        speak("Please select the location for your crop")

    def on_predict():
        try:
            loc = location_var.get()
            area = float(area_entry.get())

            if not loc:
                messagebox.showwarning("Input Required", "Please select a location.")
                return

            prod, ytph = predict(model, stats, loc, area)

            info = stats.loc[loc]

            result_text.set(
                f"🌾 Yield: {ytph:.2f} tons/ha\n"
                f"📦 Total: {prod:.2f} tons\n"
                f"📍 {loc} | {area} ha"
            )

            info_text.set(
                f"🌧 Rainfall: {info['rainfall_mm']} mm\n"
                f"🌡 Temp: {info['temperature_c']} °C\n"
                f"💧 Humidity: {info['humidity_pct']} %\n"
                f"🧪 Fertilizer: {info['fertilizer_kg_per_ha']} kg/ha"
            )

        except ValueError:
            messagebox.showerror("Invalid Input", "Enter a valid number for area.")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    root = tk.Tk()
    speak("Please select the location and area for your crop")
    root.title("Wheat Production Predictor")
    root.geometry("650x450")
    root.resizable(False, False)

    try:
        bg = Image.open("wheat_bg.jpg").resize((650, 450))
        bg_img = ImageTk.PhotoImage(bg)
        tk.Label(root, image=bg_img).place(x=0, y=0, relwidth=1, relheight=1)
    except:
        root.configure(bg="#f7f7f7")



    tk.Label(root, text="🌾 Wheat Predictor", font=("Segoe UI", 16, "bold"),
            fg="#2e7d32").pack(pady=10)

    tk.Label(root, text="Location", bg=root["bg"]).pack()
    location_var = tk.StringVar()
    def on_location_select(event):
            speak("How many area you have?")
    location_dropdown = ttk.Combobox(root, textvariable=location_var,
                                values=list(stats.index), state="readonly")
    location_dropdown.pack(pady=5)
    location_dropdown.bind("<<ComboboxSelected>>", on_location_select)

 
    tk.Label(root, text="Area (hectares)", bg=root["bg"]).pack()
    area_entry = tk.Entry(root, justify="center")
    area_entry.pack(pady=5)

    tk.Button(root, text="Predict", command=on_predict,
              bg="#4CAF50", fg="white",
              font=("Segoe UI", 10, "bold")).pack(pady=10)

    result_text = tk.StringVar()
    tk.Label(root, textvariable=result_text,
             bg="#f7f7f7", fg="#333",
             font=("Segoe UI", 11, "bold")).pack(pady=5)

    info_text = tk.StringVar()
    tk.Label(root, textvariable=info_text,
             bg=root["bg"], fg="#555",
             font=("Segoe UI", 9)).pack(pady=5)

    root.attributes("-alpha", 0.95)
    root.after(1000, greet_user)
    root.mainloop()

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "wheat_dataset_2020_2024.csv")
MODEL_PATH = os.path.join(HERE, "wheat_model.joblib")
STATS_PATH = os.path.join(HERE, "location_stats.csv")

NUM_FEATURES = [
    "area_ha", "temperature_c", "humidity_pct", "rainfall_mm",
    "soil_ph", "fertilizer_kg_per_ha", "seed_rate_kg_per_ha",
]
CAT_FEATURES = ["location", "fertilizer_type", "irrigation_type"]
TARGET = "production_tons"


def train():
    df = pd.read_csv(DATA_PATH)
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ], remainder="passthrough")

    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestRegressor(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, n_jobs=-1, random_state=42,
        )),
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"Test MAE: {mean_absolute_error(yte, pred):.2f} tons")
    print(f"Test R^2: {r2_score(yte, pred):.3f}")

    joblib.dump(model, MODEL_PATH)

    stats = (df.groupby("location")[NUM_FEATURES[1:]].mean().round(2))
    stats["fertilizer_type"] = df.groupby("location")["fertilizer_type"] \
        .agg(lambda s: s.mode().iat[0])
    stats["irrigation_type"] = df.groupby("location")["irrigation_type"] \
        .agg(lambda s: s.mode().iat[0])
    stats.to_csv(STATS_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    return model, stats


def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH):
        return joblib.load(MODEL_PATH), pd.read_csv(STATS_PATH, index_col="location")
    return train()


def predict(model, stats, location: str, area_ha: float, **overrides):
    if location not in stats.index:
        raise ValueError(f"Unknown location '{location}'. Choose from: {list(stats.index)}")
    row = stats.loc[location].to_dict()
    row.update({"location": location, "area_ha": float(area_ha)})
    row.update(overrides)
    X = pd.DataFrame([row])[NUM_FEATURES + CAT_FEATURES]
    production = float(model.predict(X)[0])
    yield_tph = production / area_ha if area_ha else 0.0
    return production, yield_tph


def interactive(model, stats):
    print("\n=== Wheat Production Predictor ===")
    print("Available locations:", ", ".join(stats.index))
    loc = input("Enter location: ").strip()
    while loc not in stats.index:
        loc = input(f"Invalid. Pick one of {list(stats.index)}: ").strip()
    area = float(input("Enter area (hectares): ").strip())
    prod, ytph = predict(model, stats, loc, area)
    print(f"\nPredicted yield : {ytph:.2f} tons/ha")
    print(f"Predicted total : {prod:.2f} tons over {area} ha in {loc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Force retrain")
    ap.add_argument("--location", type=str)
    ap.add_argument("--area", type=float)
    args = ap.parse_args()

    if args.train or not os.path.exists(MODEL_PATH):
        model, stats = train()
    else:
        model, stats = load_or_train()

    if args.location and args.area:
        prod, ytph = predict(model, stats, args.location, args.area)
        print(f"{args.location} | {args.area} ha -> {prod:.2f} tons ({ytph:.2f} t/ha)")
    else:
        launch_gui(model, stats)


if __name__ == "__main__":
    main()
