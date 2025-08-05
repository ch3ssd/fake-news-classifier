# Fake News Classifier 

This project trains a machine learning model to classify news articles as either **"Real"** or **"Fake"** based on their title and text content.

The script uses a **Logistic Regression** model and a `TfidfVectorizer` to convert text into numerical features. The final model achieves approximately 99% accuracy on the test set.

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

## ⚙️ Usage

To train the model and see the evaluation results, run the main training script from the project's root directory:

```bash
python src/train.py