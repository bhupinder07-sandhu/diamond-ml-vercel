from flask import Flask
import joblib

app = Flask(__name__)

price_model = joblib.load("models/price_model.pkl")
cut_model = joblib.load("models/cut_model.pkl")

@app.route("/")
def home():
    return "Diamond ML App is Running!"