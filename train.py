#!/usr/bin/env python


import os
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model():
    """
    A full training pipeline for the sentiment analysis model.
    """
    print("Starting model training...")

    dataset = load_dataset("imdb", split="train[:20000]")
    df = pd.DataFrame({
        'review': dataset['text'],
        'sentiment': ['positive' if label == 1 else 'negative' for label in dataset['label']]
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Take first 5000 after shuffling
    df = df.head(5000)

    print("Class distribution in sampled data:")
    print(df['sentiment'].value_counts())

    # Map sentiment to binary
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Continue with train_test_split stratified etc...


    print("Class distribution after mapping to binary:")
    print(df['sentiment'].value_counts())

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train_tfidf, y_train)

    # Save model and vectorizer
    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, "model/model.joblib")
    joblib.dump(vectorizer, "model/vectorizer.joblib")

    # Evaluate on test set
    y_pred = clf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Training complete. Test Accuracy: {acc:.2f}")

if __name__ == '__main__':
    train_model()

