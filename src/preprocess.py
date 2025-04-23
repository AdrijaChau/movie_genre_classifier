import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['Plot', 'Genre'], inplace=True)
    df['Plot'] = df['Plot'].apply(clean_text)

    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['Genre'])

    return df, le

def vectorize_text(train_texts, test_texts):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)
    return X_train, X_test, tfidf
