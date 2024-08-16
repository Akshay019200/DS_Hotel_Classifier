import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
import streamlit as st
import numpy as np
from imblearn.over_sampling import SMOTE

# Load the data
df = pd.read_csv("Hotel_Reviews.csv", encoding='latin1')

# Preprocessing
cv = CountVectorizer(max_features=5000, ngram_range=(1, 2))
x = cv.fit_transform(df["Review"]).toarray()

# Apply LSA using TruncatedSVD to reduce features to 100
lsa = TruncatedSVD(n_components=100, random_state=42)
x_reduced = lsa.fit_transform(x)

# Label Encoding
LE = LabelEncoder()
y = LE.fit_transform(df["Feedback"])

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x_reduced, y)

# Train the Logistic Regression model
logreg_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=1000)
logreg_model.fit(x_resampled, y_resampled)

# Streamlit App
st.title("Hotel Reviews Sentiment Analysis")

st.write("""
### Enter a review to predict whether it is Positive or Negative:
""")

# Input text from user
user_input = st.text_area("Review:")

# Add a Predict button
if st.button("Predict"):
    if user_input:
        # Preprocess the input
        user_input_transformed = cv.transform([user_input]).toarray()
        user_input_transformed = lsa.transform(user_input_transformed)

        # Predict the sentiment
        prediction = logreg_model.predict(user_input_transformed)
        
        # Map prediction to sentiment
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        
        # Display the result
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.write("Please enter a review to get a prediction.")
