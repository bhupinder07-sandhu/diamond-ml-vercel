from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, template_folder="../templates")

# Load Models
price_model = joblib.load("models/price_model.pkl")
cut_model = joblib.load("models/cut_model.pkl")

# Load Scalers
price_scaler = joblib.load("models/price_scaler.pkl")
cut_scaler = joblib.load("models/cut_scaler.pkl")

# Load Encoders
le_cut = joblib.load("models/le_cut.pkl")
le_color = joblib.load("models/le_color.pkl")
le_clarity = joblib.load("models/le_clarity.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_price", methods=["POST"])
def predict_price():
    return jsonify({
        "predicted_price": 12345
    })


@app.route("/predict_cut", methods=["POST"])
def predict_cut():
    return jsonify({
        "predicted_cut": "Ideal"
    })
