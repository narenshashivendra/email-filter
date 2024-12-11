import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import threading
from flask import Flask, request, jsonify
import streamlit as st
import requests

# Load the dataset
data = pd.read_csv('spam.csv')

# Preprocess and split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer and Naive Bayes classifier
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer for later use
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Flask backend setup
app = Flask(__name__)

# Load the saved model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Preprocessing function
def preprocess_email(email):
    # You can enhance this preprocessing step as per your need
    return email

@app.route('/predict', methods=['POST'])
def predict():
    email = request.json['email']
    preprocessed_email = preprocess_email(email)

    # Transform the email using the saved vectorizer
    email_vectorized = tfidf_vectorizer.transform([preprocessed_email])

    # Make the prediction
    prediction = model.predict(email_vectorized)[0]

    if prediction == 1:
        return jsonify({'prediction': 'Spam'})
    else:
        return jsonify({'prediction': 'Ham'})

# Function to run the Flask app in a separate thread
def run_flask():
    app.run(debug=True, use_reloader=False, port=5000)

# Function to handle the Streamlit frontend
def predict_email(email):
    url = 'http://localhost:5000/predict'
    data = {'email': email}
    response = requests.post(url, json=data)
    return response.json()['prediction']

# Streamlit frontend
def run_streamlit():
    st.title("Spam Email Classifier")

    email_text = st.text_area("Enter your email:")

    if st.button("Classify"):
        prediction = predict_email(email_text)
        if prediction == 'Spam':
            st.error("This is a spam email!")
        else:
            st.success("This is a ham email!")

if __name__ == "__main__":
    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Run the Streamlit frontend
    run_streamlit()
