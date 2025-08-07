import streamlit as st
import joblib
import os
from src.preprocess import clean_text

# Define paths relative to the app.py file
MODEL_PATH = os.path.join('models', 'fake_news_model.joblib')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.joblib')


@st.cache_resource
def load_model():
    """Loads the saved model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# Load the model and vectorizer when the app starts
model, vectorizer = load_model()

# --- Streamlit App Interface ---
st.title("Fake News Classifier 📰")
st.write(
    "Enter the text of a news article below to determine if it is likely to be Real or Fake."
)

if model is None or vectorizer is None:
    st.error(
        "Model files not found. Please run the training script first to generate the model."
    )
    st.code("python src/train.py")
else:
    # Best Practice: Add instructions for the user
    st.info("For the most accurate prediction, please paste the full article text, including the title.")

    # Create a text area for user input
    user_input = st.text_area("Article Text", height=250, placeholder="Paste your article here...")

    # Create a button to trigger the classification
    if st.button("Classify"):
        if user_input:
            # 1. Clean the user's input text
            cleaned_input = clean_text(user_input)

            # 2. Vectorize the cleaned text
            vectorized_input = vectorizer.transform([cleaned_input])

            # 3. Make a prediction and get probabilities
            prediction = model.predict(vectorized_input)
            probability = model.predict_proba(vectorized_input)

            # 4. Display the result
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                confidence = probability[0][1] * 100
                st.success(f"This article is likely REAL (Confidence: {confidence:.2f}%)")
            else:
                confidence = probability[0][0] * 100
                st.error(f"This article is likely FAKE (Confidence: {confidence:.2f}%)")
        else:
            st.warning("Please enter some text to classify.")