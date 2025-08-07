# Fake News Classifier 

This project trains a machine learning model to classify news articles and deploys it as an interactive web application using Streamlit.

The script uses a **Logistic Regression** model and a `TfidfVectorizer`. The final model achieves approximately 99% accuracy on the test set.

---

## Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

- Python 3.8+
- The dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

### Installation

1.  **Clone the repository:**
    *Remember to replace the URL with your own GitHub repository link.*
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Place the data:**
    - Download the dataset from the link above.
    - Create a `data` folder inside the project directory.
    - Place `Fake.csv` and `True.csv` inside this `data` folder.

3.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python3 -m venv venv

    # Activate it (on macOS/Linux)
    source venv/bin/activate

    # Activate it (on Windows)
    # venv\Scripts\activate
    ```

4.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

There are two main steps to use this project.

### Step 1: Train the Model

First, you must run the training pipeline. This script will process the data and save the trained model and vectorizer to a `models/` folder.

```bash
python src/train.py
```

### Step 2: Run the Web Application

Once the model is trained and saved, you can launch the interactive Streamlit app.

```bash
streamlit run app.py
```
This will open a new tab in your web browser where you can paste any news article text and get a prediction.
