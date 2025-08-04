import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
df = pd.read_csv('navie-bayes/spam.csv', encoding='latin-1')


def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


df['clean_text'] = df['Message'].apply(clean_text)
df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
 
 
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


def predict_message(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "SPAM" if pred == 1 else "HAM (oddiy xabar)"

while True:
    user_input = input("\nXabar kiriting (chiqish uchun 'exit'): ")
    if user_input.lower() == 'exit':
        break
    result = predict_message(user_input)
    print("Natija:", result)
