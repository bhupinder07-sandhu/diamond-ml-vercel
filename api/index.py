from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, template_folder="../templates")

price_model = joblib.load("models/price_model.pkl")
cut_model = joblib.load("models/cut_model.pkl")

price_scaler = joblib.load("models/price_scaler.pkl")
cut_scaler = joblib.load("models/cut_scaler.pkl")

le_cut = joblib.load("models/le_cut.pkl")
le_color = joblib.load("models/le_color.pkl")
le_clarity = joblib.load("models/le_clarity.pkl")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/price")
def price():
    return render_template("price.html")

@app.route("/cut")
def cut():
    return render_template("cut.html")



@app.route("/predict_price", methods=["POST"])
def predict_price():
    try:
        data = request.get_json()

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

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

@app.route("/predict_cut", methods=["POST"])
def predict_cut():
    try:
        data = request.get_json()

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

        cut_name = le_cut.inverse_transform([int(prediction[0])])[0]

        return jsonify({
            "predicted_cut": cut_name
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })
