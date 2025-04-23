import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['plot', 'genre'], inplace=True)
    df['plot'] = df['plot'].apply(clean_text)

    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['genre'])

    return df, le

def vectorize_text(train_texts, test_texts):
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)
    return X_train, X_test, tfidf
