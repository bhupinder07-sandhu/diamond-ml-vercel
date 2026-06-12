from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="../templates")

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
