from flask import Flask, render_template,request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method=="POST":
        msg=request.form.get("message")
        print(msg)
    else:
        return render_template("predict.html")
      
if __name__ == '__main__':
    app.run(debug=True,host = '0.0.0.0',port = 6060) 