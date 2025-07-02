import joblib

# Load the trained model
model = joblib.load('language_detection_model.pkl')

print("✅ Model loaded successfully!")

# Sample texts to test
sample_texts = [
   
    "Hola, ¿cómo estás?",
    
]

# Predict languages
predictions = model.predict(sample_texts)

for text, lang in zip(sample_texts, predictions):
    print(f"Text: {text}")
    print(f"Predicted Language: {lang}")
    print("---")
