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
    return render_template("index.html")



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
    return jsonify({
        "predicted_cut": "Ideal"
    })
