import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('language_detection_model.pkl')

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Urdu": "ur",
    "Persian": "fa"
}

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/detect_translate', methods=['POST'])
def detect_and_translate():
    data = request.get_json()
    text = data.get('text', '')
    target_lang_name = data.get('target_lang', 'English')
    target_lang_code = LANGUAGES.get(target_lang_name, 'en')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        detected_lang = model.predict([text])[0]
        translation = GoogleTranslator(source=detected_lang, target=target_lang_code).translate(text)
        return jsonify({
            'detected_lang': detected_lang,
            'translation': translation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
