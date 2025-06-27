
# Sentiment Analysis ML Pipeline

NOTE: I LOVE VIBE CODING, AND I USED IT OBVIOUSLY!!

This project demonstrates a basic machine learning pipeline for sentiment analysis on text data using scikit-learn.

## How to Install Dependencies

First, ensure you have Python 3 installed. Then, create a virtual environment and install the required packages.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```

## How to Train the Model

The `train.py` script handles the complete training pipeline: loading data, vectorizing it, training the model, and saving the artifacts.

```bash
# Run the training script
python train.py
```

This will create two files in the `./model` directory: `vectorizer.joblib` and `model.joblib`.

## How to Run Predictions

Use the `predict.py` script to classify a new review. Pass the review text as a command-line argument.

```bash
# Example of predicting a positive review
python predict.py "This was an absolutely fantastic film with brilliant acting!"

# Example of predicting a negative review
python predict.py "I was really bored and the plot was predictable."
```
## How to use the API

Use the following curl command

```bash
# Example of using CURL
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"review": "I loved this movie, it was fantastic!"}'

# Sample output
{
  "confidence": 0.7729591168616868,
  "prediction": "positive"
}
```

The script will output the predicted sentiment ('positive' or 'negative') and a confidence score.

