from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load the saved model and vectorizer from Task 2
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("sentiment_vectorizer.joblib")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sentiment API is running. Send a POST request to /predict"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please send JSON like {\"text\": \"your sentence\"}"}), 400

    cleaned = clean_text(data["text"])
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    return jsonify({"text": data["text"], "sentiment": prediction})


if __name__ == "__main__":
    app.run(port=5000)