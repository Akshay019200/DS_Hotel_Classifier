import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from imblearn.over_sampling import RandomOverSampler

# Load and preprocess the data (caching data transformations)
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("Hotel_Reviews.csv", encoding='latin1')

    # Text vectorization with bigrams
    cv = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    x = cv.fit_transform(df["Review"]).toarray()

    # Label encoding
    LE = LabelEncoder()
    y = LE.fit_transform(df["Feedback"])

    # Resampling to handle class imbalance
    ros = RandomOverSampler(random_state=42)
    x_resample, y_resample = ros.fit_resample(x, y)

    # Apply LSA using TruncatedSVD for dimensionality reduction
    n_components = 100  # Number of dimensions to reduce to
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    x_lsa = lsa.fit_transform(x_resample)

    return cv, lsa, x_lsa, y_resample

# Define and compile the ANN model (without caching due to TensorFlow serialization issues)
def create_ann_model(input_dim):
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=100, activation="relu", input_dim=input_dim))
    ann.add(tf.keras.layers.Dense(units=150, activation="relu"))
    ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    ann.compile(optimizer=tf.keras.optimizers.Nadam(), loss="binary_crossentropy", metrics=["accuracy"])
    return ann

# Train the model (without caching due to TensorFlow serialization issues)
def train_ann_model(x_lsa, y_resample):
    ann = create_ann_model(x_lsa.shape[1])
    ann.fit(x_lsa, y_resample, epochs=50, batch_size=32, verbose=0)
    return ann

# Load preprocessed data
cv, lsa, x_lsa, y_resample = load_and_preprocess_data()

# Train the model
ann_model = train_ann_model(x_lsa, y_resample)

# Streamlit App
st.title("Hotel Review Sentiment Analysis")

st.write("""
### Predict if a review is Positive or Negative
Enter a review below:
""")

# Input text
review_input = st.text_area("Review", "")

# Prediction button
if st.button("Predict Sentiment"):
    if review_input.strip() != "":
        # Transform input review
        review_vector = cv.transform([review_input]).toarray()
        review_lsa = lsa.transform(review_vector)
        
        # Make prediction
        prediction = ann_model.predict(review_lsa)
        sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
        
        st.write(f"The review is: **{sentiment}**")
    else:
        st.write("Please enter a review to predict.")
