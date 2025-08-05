import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text, download_nltk_data
import joblib
import os

print("Loading data...")
download_nltk_data()

df_fake = pd.read_csv('data/Fake.csv')
df_true = pd.read_csv('data/True.csv')
df_fake['label'] = 0
df_true['label'] = 1
df = pd.concat([df_fake, df_true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Data loaded and combined.")

print("Cleaning and preprocessing text... (This may take a few minutes)")
df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
df['cleaned_text'] = df['full_text'].apply(clean_text)
print("Text processing complete.")

print("Splitting data and creating features...")
X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Features created.")

print("Training the Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

print("Evaluating the model...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Fake', 'True'])

print("\n--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("--------------------------------\n")

# Define the directory to save models
model_dir = 'models'
# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Define file paths
model_path = os.path.join(model_dir, 'fake_news_model.joblib')
vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')

# Save the trained model and the vectorizer to disk
print(f"Saving model to {model_path}")
joblib.dump(model, model_path)

print(f"Saving vectorizer to {vectorizer_path}")
joblib.dump(vectorizer, vectorizer_path)

print("Model and vectorizer saved successfully.")