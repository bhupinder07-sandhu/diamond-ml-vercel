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

    data = request.json

    features = [[
        le_cut.transform([data["cut"]])[0],
        float(data["carat"]),
        le_color.transform([data["color"]])[0],
        le_clarity.transform([data["clarity"]])[0],
        float(data["depth"]),
        float(data["table"]),
        float(data["x"]),
        float(data["y"]),
        float(data["z"])
    ]]

    features = price_scaler.transform(features)

    prediction = price_model.predict(features)

    return jsonify({
        "predicted_price": round(float(prediction[0]), 2)
    })


@app.route("/predict_cut", methods=["POST"])
def predict_cut():

    data = request.json

    features = [[
        float(data["carat"]),
        le_color.transform([data["color"]])[0],
        le_clarity.transform([data["clarity"]])[0],
        float(data["depth"]),
        float(data["table"]),
        float(data["price"]),
        float(data["x"]),
        float(data["y"]),
        float(data["z"])
    ]]

    features = cut_scaler.transform(features)

    prediction = cut_model.predict(features)

    cut_name = le_cut.inverse_transform(prediction)[0]

    return jsonify({
        "predicted_cut": str(cut_name)
    })
