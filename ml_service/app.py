from flask import Flask, request, jsonify
import sys
import json

# 👇 import your existing model code
# from predict import predict_text   (example)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text missing"}), 400

    # 👇 CALL YOUR REAL MODEL HERE
    result = predict_text(text)   # must return dict

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)