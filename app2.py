import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Title and description
st.title('Fake News Detection')
st.write('This is a simple machine learning app to detect fake news.')

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe:", df.head())
    
    # Get shape and display
    st.write(f"Dataset shape: {df.shape}")
    
    # Get the labels
    labels = df['label']
    st.write("Labels:", labels.head())
    
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
    
    # Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)
    
    # Initialize a PassiveAggressiveClassifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    # Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {round(score*100, 2)}%')
    
    # Build confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    st.write("Confusion Matrix:")
    st.write(cm)

