from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer once when starting the server
model = joblib.load("model/model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review')
    if not review:
        return jsonify({'error': 'No review text provided'}), 400

    X = vectorizer.transform([review])
    pred_proba = model.predict_proba(X)[0]
    pred_class = model.predict(X)[0]

    label = 'positive' if pred_class == 1 else 'negative'
    confidence = float(pred_proba[pred_class])

    return jsonify({'prediction': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)

