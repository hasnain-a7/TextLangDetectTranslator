import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('dataset.csv')

# ✅ Drop any empty rows to avoid NaNs
df = df.dropna()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['lang'], test_size=0.2, random_state=42
)

# Create pipeline: vectorizer + classifier
model = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1,3), analyzer='char')),
    ('classifier', MultinomialNB())
])

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'language_detection_model.pkl')
print("✅ Model trained and saved as language_detection_model.pkl")
