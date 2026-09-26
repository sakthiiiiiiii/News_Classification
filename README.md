# News Classification

An NLP and machine learning project for classifying news articles into categories such as sports, politics, and other news categories.

## Project Overview

This project processes news articles, extracts meaningful text features, trains a machine learning model, and predicts the category of new articles.

The project includes:

- News data loading and preprocessing
- Text cleaning and lemmatization
- TF-IDF feature extraction
- News category classification
- Model training and evaluation
- Command-line predictions
- FastAPI-based prediction service
- Category-based news feed storage

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- NLTK
- spaCy
- Regex
- Joblib
- FastAPI
- Uvicorn
- Matplotlib
- Seaborn
- imbalanced-learn

## Project Structure

```text
News_Classification/
├── data/
│   ├── feeds/
│   ├── processed/
│   └── raw/
├── models/
│   ├── news_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── process_dataset.py
│   ├── features/
│   │   └── vectorizer.py
│   ├── models/
│   │   ├── build_model.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train.py
│   └── routing/
│       └── store.py
├── main.py
├── requirements.txt
└── README.md
```

## How the Project Works

The project follows these steps:

1. Load raw news data from the `data/raw/` directory.
2. Combine the news headline and article text.
3. Convert the text to lowercase.
4. Remove URLs and unwanted characters.
5. Remove stop words using spaCy.
6. Lemmatize the text.
7. Convert the cleaned text into TF-IDF features.
8. Split the data into training and testing sets.
9. Balance the training data using random oversampling.
10. Train a `LinearSVC` classification model.
11. Save the trained model and vectorizer.
12. Predict categories for new articles.

## Dataset

Raw dataset files should be placed inside:

```text
data/raw/
```

The data loader searches for files using the following pattern:

```text
data/raw/inshort_news_data-*.csv
```

The dataset is expected to contain the following columns:

| Column | Description |
|---|---|
| `news_headline` | Headline of the news article |
| `news_article` | Main article content |
| `news_category` | Category assigned to the article |

## Installation

Clone the repository:

```bash
git clone https://github.com/sakthiiiiiiii/News_Classification.git
cd News_Classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the required spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

### 1. Process the Dataset

Run the main script:

```bash
python main.py
```

This processes the raw news files and creates:

```text
data/processed/cleaned_news.csv
```

### 2. Generate TF-IDF Features

Run the vectorizer:

```bash
python src/features/vectorizer.py
```

The trained vectorizer is saved as:

```text
models/tfidf_vectorizer.pkl
```

The vectorizer uses:

- A maximum of 20,000 features
- Unigrams and bigrams

### 3. Train the Model

Run the training script:

```bash
python src/models/train.py
```

The trained model is saved as:

```text
models/news_model.pkl
```

The project uses a `LinearSVC` model for classification.

### 4. Evaluate the Model

Run:

```bash
python src/models/evaluate.py
```

The evaluation script displays:

- Classification report
- Confusion matrix
- Category-wise prediction results

### 5. Make Command-Line Predictions

Run:

```bash
python src/models/predict.py
```

Enter a news article when prompted:

```text
Enter News: The team won the final match
Prediction: sports
```

## FastAPI Service

The project also provides a FastAPI service for making predictions.

Start the API with:

```bash
uvicorn src.api.app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Prediction API

### Endpoint

```http
POST /predict
```

### Request

```json
{
  "text": "The team won the championship after a close final match."
}
```

### Response

```json
{
  "category": "sports"
}
```

Each predicted article is also stored in a category-specific CSV file inside:

```text
data/feeds/
```

## Model Information

### Feature Extraction

The project uses `TfidfVectorizer` to convert text into numerical features.

```python
TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)
```

### Classification Model

The classifier is a `LinearSVC` model:

```python
LinearSVC(C=1.0)
```

### Class Balancing

`RandomOverSampler` is used to balance the training data before model training.

## Important Notes

- Run the commands from the project root directory.
- Install the `en_core_web_sm` spaCy model before running preprocessing.
- Raw files must follow the expected filename pattern.
- The dataset must contain `news_headline`, `news_article`, and `news_category` columns.
- The model and vectorizer files are loaded from the `models/` directory.
- Retrain the model when the dataset or preprocessing code changes.

## Future Improvements

- Add more news categories.
- Add a web-based user interface.
- Improve model accuracy through hyperparameter tuning.
- Add automated tests.
- Add model performance visualizations.
- Add API validation and error handling.
- Deploy the FastAPI service.

## License

No license has currently been added to this project.
