import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def download_nltk_data():
    """Checks for NLTK data and downloads it if not found."""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("Downloading NLTK data (stopwords, wordnet, omw-1.4)...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        print("NLTK data downloaded.")

def clean_text(text):
    """Cleans and preprocesses raw text."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.lower().split()
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)