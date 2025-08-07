import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Import path configurations and preprocessing functions
from preprocess import clean_text, download_nltk_data
from config import FAKE_CSV_PATH, TRUE_CSV_PATH, MODEL_PATH, VECTORIZER_PATH, MODELS_DIR

# --- 1. Load Data ---
print("Loading data...")
download_nltk_data()

df_fake = pd.read_csv(FAKE_CSV_PATH)
df_true = pd.read_csv(TRUE_CSV_PATH)
df_fake['label'] = 0
df_true['label'] = 1
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data loaded and combined.")

# --- 2. Preprocess Text ---
print("Cleaning and preprocessing text... (This may take a few minutes)")
df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
df['cleaned_text'] = df['full_text'].apply(clean_text)
print("Text processing complete.")

# --- 3. Create Features and Split Data ---
print("Splitting data and creating features...")
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Features created.")

# --- 4. Train Model ---
print("Training the Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- 5. Evaluate Model ---
print("Evaluating the model...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Fake', 'True'])

print("\n--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("--------------------------------\n")

# --- 6. Save Model and Vectorizer ---
# Create the models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Save the trained model and the vectorizer to disk
print(f"Saving model to {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print(f"Saving vectorizer to {VECTORIZER_PATH}")
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model and vectorizer saved successfully.")