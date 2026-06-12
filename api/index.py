from flask import Flask
import joblib

app = Flask(__name__)

price_model = joblib.load("models/price_model.pkl")
cut_model = joblib.load("models/cut_model.pkl")

price_scaler = joblib.load("models/price_scaler.pkl")
cut_scaler = joblib.load("models/cut_scaler.pkl")

le_cut = joblib.load("models/le_cut.pkl")
le_color = joblib.load("models/le_color.pkl")
le_clarity = joblib.load("models/le_clarity.pkl")

@app.route("/")
def home():
    return "All files loaded successfully!"
