import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import string
import nltk
from nltk.corpus import stopwords

# NLTK stopwords (birinchi marta ishlatsangiz, quyidagini uncomment qiling)
# nltk.download('stopwords')

# 1. Ma’lumotni o‘qish
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['category', 'message']]
df.columns = ['label', 'text']

# 2. Tozalash funksiyasi
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# 3. Labelni raqamlash (spam=1, ham=0)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Train/Test ajratish
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label_num'], test_size=0.2, random_state=42)

# 5. TF-IDF vektorlash
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Modelni o‘qitish
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Bashorat va natijalar
y_pred = model.predict(X_test_vec)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



# 8. Kiritilgan matnga qarab spam/ham aniqlash funksiyasi
def predict_message(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "📨 SPAM" if pred == 1 else "✅ HAM (oddiy xabar)"

# 🔎 Sinab ko‘ramiz
while True:
    user_input = input("\n✉️ Xabar kiriting (chiqish uchun 'exit'): ")
    if user_input.lower() == 'exit':
        break
    result = predict_message(user_input)
    print("➡️ Natija:", result)
