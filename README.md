Project: Sentiment Analysis on Reviews

This project performs sentiment analysis on customer reviews and exposes functionality via an API. It includes data exploration, model training, and serving predictions.

📂 Project Structure

Reviews.csv → Dataset containing text reviews.

Sentiment-analysis.ipynb → Main notebook for training and analyzing sentiments.

Sentiment-analysis-tests.ipynb → Notebook for testing model predictions.

sentment.py → Core Python script for sentiment classification logic.

api_code.py → Exposes the sentiment model as an API (so other apps can call it).

downloader.py → Helper script for fetching datasets or dependencies.

test.py → Additional test cases.

Figure_1.png → A generated graph (likely showing results such as word frequency, accuracy, or sentiment distribution).

🔧 Workflow

Data Loading
Reads reviews from Reviews.csv.

Preprocessing

Tokenization

Stopword removal

Lowercasing, stemming/lemmatization

Model Training

Machine Learning model (likely Logistic Regression, Naive Bayes, or similar).

Trains on labeled reviews for positive/negative sentiment.

Evaluation

Accuracy, precision, recall, F1-score.

Visual results stored in Figure_1.png.

API Exposure

Using api_code.py, you can send text to the API and get sentiment back (positive, negative, maybe neutral).
