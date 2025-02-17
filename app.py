from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

app = Flask(__name__)

# Load vectorizer and model once at startup
with open("vectorizer.pickle", "rb") as vc:
    vectorizer = pickle.load(vc)

with open("model.pickle", "rb") as mc:
    model = pickle.load(mc)

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

        # Make prediction
        pred = model.predict(vcdata)
        print(f"Prediction: {pred}")

        return jsonify({"prediction": pred.tolist()}), 200

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)