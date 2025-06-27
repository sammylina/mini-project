#!/usr/bin/env python

import sys
import joblib

if len(sys.argv) != 2:
    print("Usage: python predict.py \"<review_text>\"")
    sys.exit(1)

review = sys.argv[1]

# Load model and vectorizer
clf = joblib.load("model/model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")

# Transform and predict
X = vectorizer.transform([review])
proba = clf.predict_proba(X)[0]
pred = clf.predict(X)[0]

label = "positive" if pred == 1 else "negative"
confidence = proba[pred]

print(f"{label} ({confidence:.2f})")

