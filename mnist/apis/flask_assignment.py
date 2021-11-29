from flask import Flask
from flask import request
import numpy as np
import joblib


app = Flask(__name__)


best_model_path_svm = '../models/tt_0.15_val_0.15_rescale_0.5_gamma_0.0001/model_info.joblib'


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/svm_predict',methods=['POST'])
def predict():
    clf = joblib.load(best_model_path_svm)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    print(str(predicted[0]))
    return str(predicted[0])

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8000)