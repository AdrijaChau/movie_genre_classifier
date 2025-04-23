# movie_genre_classifier

 Objective
Develop a machine learning model to classify movies into genres based on their plot descriptions using text processing and classification algorithms.

 Features
- Text preprocessing and cleaning
- TF-IDF vectorization
- Multiple classifiers (Naive Bayes, Logistic Regression, Random Forest)
- Model evaluation and reporting

1. Prerequisites

Python 3.7+
Git
pip (Python package installer)

2. Steps to run the code

i.Clone the Repository
Open a terminal and run:
git clone https://github.com/your-username/movie-genre-classifier.git cd movie-genre-classifier2

ii. Install dependencies:
pip install -r requirements.txt

iii. Download the Dataset
Go to the dataset page: IMDb Genre Classification on Kaggle
Download IMDB Dataset.csv
Place it inside the data/ folder

iv. Train the Model
Run the training script:
python src/train.py

This script:

Cleans and vectorizes the plot descriptions
Trains Naive Bayes, Logistic Regression, and Random Forest models
Prints classification reports for each
Saves the best model (Logistic Regression by default) and preprocessing artifacts (TF-IDF vectorizer and label encoder) in the models/ folder

3. Check Outputs
We can view the classification performance in the terminal
