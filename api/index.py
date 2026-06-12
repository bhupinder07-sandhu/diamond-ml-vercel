from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Models
price_model = joblib.load("models/price_model.pkl")
cut_model = joblib.load("models/cut_model.pkl")

# Scalers
price_scaler = joblib.load("models/price_scaler.pkl")
cut_scaler = joblib.load("models/cut_scaler.pkl")

# Encoders
le_cut = joblib.load("models/le_cut.pkl")
le_color = joblib.load("models/le_color.pkl")
le_clarity = joblib.load("models/le_clarity.pkl")


@app.route("/")
def home():
    return "Diamond ML App Running Successfully!"


@app.route("/predict_price", methods=["POST"])
def predict_price():

    data = request.json

    cut = le_cut.transform([data["cut"]])[0]
    color = le_color.transform([data["color"]])[0]
    clarity = le_clarity.transform([data["clarity"]])[0]

    features = [[
        cut,
        data["carat"],
        color,
        clarity,
        data["depth"],
        data["table"],
        data["x"],
        data["y"],
        data["z"]
    ]]

    features = price_scaler.transform(features)

    prediction = price_model.predict(features)

    return jsonify({
        "predicted_price": float(prediction[0])
    })


@app.route("/predict_cut", methods=["POST"])
def predict_cut():

    data = request.json

    color = le_color.transform([data["color"]])[0]
    clarity = le_clarity.transform([data["clarity"]])[0]

    features = [[
        data["carat"],
        color,
        clarity,
        data["depth"],
        data["table"],
        data["price"],
        data["x"],
        data["y"],
        data["z"]
    ]]

    features = cut_scaler.transform(features)

    prediction = cut_model.predict(features)

    cut_name = le_cut.inverse_transform(prediction)[0]

    return jsonify({
        "predicted_cut": str(cut_name)
    })
