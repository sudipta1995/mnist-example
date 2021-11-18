from flask import Flask, request
import numpy as np
import sys
sys.path.insert(1, '/home/sudipta/mlops/mnist-example/mnist-example')
import utils

app = Flask(__name__)
clf = utils.load('/home/sudipta/mlops/mnist-example/mnist-example/model.pkl')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return str(predicted[0])

app.run('0.0.0.0', debug = True, port = '5000')