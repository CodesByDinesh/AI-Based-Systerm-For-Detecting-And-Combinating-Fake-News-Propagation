import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and prepare dataset
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\final project\dataset1.csv", encoding='ISO-8859-1', header=None, names=['text', 'label'])
data.dropna(subset=['text', 'label'], inplace=True)
data['text'] = data['text'].astype(str)

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

accuracy = accuracy_score(y_test, model.predict(vectorizer.transform(X_test)))
print(f"Accuracy on test data: {accuracy * 100:.2f}%")


def predict_news(text):
    tfidf = vectorizer.transform([text])
    prediction = model.predict(tfidf)[0]
    return "REAL" if prediction == 1 else "FAKE"
