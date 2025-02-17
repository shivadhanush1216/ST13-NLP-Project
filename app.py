from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

app = Flask(__name__)

# Check if vectorizer exists, else create and save one
VECTOR_PATH = "vectorizer.pickle"
MODEL_PATH = "model.pickle"

if not os.path.exists(VECTOR_PATH):
    print("Vectorizer not found! Training and saving a new one...")
    sample_texts = ["This is an example", "Text processing in NLP", "Machine learning and AI"]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(sample_texts)
    
    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print("New vectorizer saved successfully!")

# Load vectorizer and model
with open(VECTOR_PATH, "rb") as vc:
    vectorizer = pickle.load(vc)

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as mc:
        model = pickle.load(mc)
else:
    model = None  # Handle the case where the model is missing

# Text preprocessing class
class TextToNum:
    def __init__(self, text):
        self.text = text

    def cleaner(self):
        # Remove special characters and numbers
        self.text = re.sub(r'[^a-zA-Z\s]', '', self.text)
        self.text = self.text.lower()

    def token(self):
        # Tokenize the text
        self.text = self.text.split()

    def removeStop(self):
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        self.text = [word for word in self.text if word not in stop_words]

    def stemme(self):
        # Stem the words
        ps = PorterStemmer()
        self.text = [ps.stem(word) for word in self.text]
        return self.text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")
        
        if not msg:
            return jsonify({"error": "No input text provided"}), 400
        
        print(f"Received message: {msg}")
        
        # Process text
        ob = TextToNum(msg)
        ob.cleaner()
        ob.token()
        ob.removeStop()
        st = ob.stemme()
        stem_vector = " ".join(st)
        print(f"Stemmed Vector: {stem_vector}")

        # Ensure vectorizer is valid
        if not hasattr(vectorizer, "transform"):
            return jsonify({"error": "Invalid vectorizer object"}), 500

        # Vectorize input text
        vcdata = vectorizer.transform([stem_vector]).toarray()
        print(f"Vectorized Data: {vcdata}")

        # Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model file missing"}), 500

        # Make prediction
        pred = model.predict(vcdata)
        print(f"Prediction: {pred}")

        return jsonify({"prediction": pred.tolist()}), 200

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
