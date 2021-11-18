from flask import Flask
from flask.globals import request
from mnist.utils 
import numpy as np

app = Flask(__name__)

best_model_path = '.mnist/models/model.joblib'
clf = load(best_model_path)

@app.route("/")
def hello_world():
    return "<p>Hello </p>"


@app.route("/predict", methods=['POST'])
def predict():
    print("Hello")
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return "<p>Hello Predict</p>"

if __name__ == "main":
    app.run()
