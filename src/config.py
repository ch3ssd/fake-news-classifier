import os

# Get the absolute path of the directory where this file is located (the 'src' directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the project's root directory (which is one level up from 'src')
ROOT_DIR = os.path.dirname(BASE_DIR)

# Define paths for data and models relative to the root directory
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Define the full paths to your specific files
FAKE_CSV_PATH = os.path.join(DATA_DIR, 'Fake.csv')
TRUE_CSV_PATH = os.path.join(DATA_DIR, 'True.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'fake_news_model.joblib')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')