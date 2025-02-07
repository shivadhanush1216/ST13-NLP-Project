from flask import Flask, render_template
app=Flask(__name__)

@app.route('/')
# API functoion
def index():
    return render_template('index.html')
@app.route('/predict')
# API functoion
def predict():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5050,debug=True)
