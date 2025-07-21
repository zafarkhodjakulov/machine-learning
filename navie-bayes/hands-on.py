from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv(r'navie-bayes/spam.csv', encoding='latin-1')

df['label'] = df['Category'].map({'ham': 0, 'spam': 1})

x = df['Message']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)


model = MultinomialNB()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))   
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def predict_message(text):
    text_vec = vectorizer.transform([text])
    y_p = model.predict(text_vec)
    return "📨 SPAM" if y_p[0] == 1 else "✅ HAM (oddiy xabar)"
    
prd = predict_message("You have two cars and a house in your name.")
print(prd)


# def predict_msg(text):
#     cleaned = text.lower()
#     cleaned = ''.join([char for char in cleaned if char not in string.punctuation])
#     words = cleaned.split()
#     stop_words = stopwords.words('english')
#     words = [word for word in words if word not in stop_words]
#     cleaned_text = ' '.join(words)
    
#     vec = vectorizer.transform([cleaned_text])
#     pred = model.predict(vec)[0]
#     return "📨 SPAM" if pred == 1 else "✅ HAM (oddiy xabar)"


