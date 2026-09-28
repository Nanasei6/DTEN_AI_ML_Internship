# Task 3: Sentiment Analysis Web API

A lightweight Flask API that wraps the airline-tweet sentiment model from Task 2
(TF-IDF + Logistic Regression with balanced class weights). It accepts a sentence
and returns a predicted sentiment: positive, negative or neutral.

## How to run

```
conda activate dten-ai
pip install flask scikit-learn joblib
python app.py
```

The API runs at http://127.0.0.1:5000

## Files

- `app.py` : the Flask application
- `sentiment_model.joblib` : the trained model (from Task 2)
- `sentiment_vectorizer.joblib` : the fitted TF-IDF vectorizer (from Task 2)

## Endpoints

### GET /
Health check. Confirms the API is running.

Response:
```
{"message": "Sentiment API is running. Send a POST request to /predict"}
```

### POST /predict
Accepts JSON with a `text` field and returns the predicted sentiment.

**Example request 1 (positive):**
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"Amazing crew, the flight was smooth and on time!\"}"
```
Response:
```
{"sentiment":"positive","text":"Amazing crew, the flight was smooth and on time!"}
```

**Example request 2 (negative):**
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\": \"Worst airline ever, my bag was lost and nobody helped.\"}"
```
Response:
```
{"sentiment":"negative","text":"Worst airline ever, my bag was lost and nobody helped."}
```

**Example request 3 (invalid input, missing "text" field):**
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"wrong\": \"oops\"}"
```
Response (HTTP 400):
```
{"error":"Please send JSON like {\"text\": \"your sentence\"}"}
```

## Model accuracy

On the 2,928-tweet test set from Task 2, the model achieves 74.25% accuracy
(macro F1 of 0.70). Recall per class: negative 0.77, neutral 0.69, positive 0.68.

## Limitations

1. **Neutral text is often misclassified.** The training data is airline tweets,
   which are mostly complaints. A factual sentence such as "Flight departs at 6pm
   from gate 12." was predicted as negative, because words like "flight" and "gate"
   appear mostly in negative tweets. The model counts words and does not understand
   context or sarcasm.
2. **Narrow domain.** It was trained only on airline tweets and may perform poorly
   on other kinds of text.
3. **Development server.** Flask's built-in server is for testing only. A real
   deployment would use a production server such as Gunicorn.


   