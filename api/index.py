from flask import Flask
import joblib

app = Flask(__name__)

price_model = joblib.load("models/price_model.pkl")

@app.route("/")
def home():
    return "Price model loaded successfully!"
