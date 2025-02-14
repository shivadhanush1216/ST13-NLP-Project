from flask import Flask, render_template, request, jsonify
from test import TextToNum
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        msg = request.form.get("message")
        print("Original Message:", msg)

        # Preprocess the text
        ob = TextToNum(msg)
        ob.cleaner()
        print("Cleaned Text:", ob.cleaned)

        ob.token()
        print("Tokenized Words:", ob.tkns)

        ob.removeStop()
        print("After Removing Stopwords:", ob.cl)

        st = ob.stemme()
        print("After Stemming:", st)

        stem_vector = " ".join(st)
        print("Final Processed Text for Vectorization:", stem_vector)

        # Load Vectorizer
        try:
            with open("vectorizer.pickle", "rb") as vc:
                vectorizer = pickle.load(vc)
            vcdata = vectorizer.transform([stem_vector]).toarray()
            print("Vectorized Data:", vcdata)
        except Exception as e:
            print("Error loading vectorizer:", e)
            return jsonify({"error": "Vectorizer loading failed"})

        # Load Model
        try:
            with open("model.pickle", "rb") as mc:
                model = pickle.load(mc)
            pred = model.predict(vcdata)
            print("Predicted Sentiment:", pred[0])
        except Exception as e:
            print("Error loading model:", e)
            return jsonify({"error": "Model loading failed"})

        return render_template("result.html", sentiment=str(pred[0]))

    else:
        return render_template("predict.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5050')
