import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import prepare_data, vectorize_text

def train_models():
    df, le = prepare_data("data/IMDB Dataset.csv")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        df['Plot'], df['genre_encoded'], test_size=0.2, random_state=42)

    X_train, X_test, tfidf = vectorize_text(X_train_texts, X_test_texts)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier()
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, preds, target_names=le.classes_))

    # Save best model (or all models if preferred)
    joblib.dump(models["Logistic Regression"], "models/saved_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

if __name__ == "__main__":
    train_models()

