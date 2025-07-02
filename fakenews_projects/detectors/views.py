import pandas as pd
from django.shortcuts import render, redirect
from django.core.mail import send_mail
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset once at server start
data = pd.read_csv(r"C:\Users\LENOVO\Desktop\final project\dataset1.csv",
                   encoding='ISO-8859-1', header=None,
                   names=['text', 'label'])
data.dropna(subset=['text', 'label'], inplace=True)
data['text'] = data['text'].astype(str)

X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

existing_news_set = set(data['text'].str.strip().str.lower())

def predict_news(text):
    tfidf = vectorizer.transform([text])
    prediction = model.predict(tfidf)[0]
    return "REAL" if prediction == 1 else "FAKE"

def get_email(request):
    message = None
    success = False

    if request.method == "POST":
        email = request.POST.get('email', '').strip()
        if email and '@' in email:
            # Save email to session
            request.session['user_email'] = email
            message = "Thank you for subscribing!"
            success = True
            # Redirect to analyze page after successful subscription
            return redirect('index')
        else:
            message = "Invalid email address, please try again."
            success = False

    return render(request, 'detectors/email_input.html',
                  {'message': message, 'success': success})

def index(request):
    result = None
    error = None

    if request.method == 'POST':
        news = request.POST.get('news', '').strip()
        user_email = request.session.get('user_email')

        if not user_email:
            error = "Please subscribe with your email to proceed further."
        elif not news:
            error = "Please enter some news text."
        elif news.lower() not in existing_news_set:
            error = "🚫 This news article is Invalid!"
        else:
            result = predict_news(news)
            # Send email to user with result
            try:
                send_mail(
                    subject="Your Fake News Detection Result",
                    message=f"Hello User,Thank you for using our Fake News Detection tool. We have carefully analyzed the content you submitted using our AI-based detection system, and the results are now available. The news you submitted is classified as: {result}",
                    from_email=None,
                    recipient_list=[user_email],
                    fail_silently=False,
                )
            except Exception as e:
                error = f"Failed to send email: {str(e)}"

    return render(request, 'detectors/index.html', {'result': result, 'error': error})

    
