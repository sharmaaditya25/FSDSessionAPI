import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys

sys.path.append('../1.Development/')
# os.chdir('../2.Deployment/')

from config import modeloutput

# Create flask app
flask_app = Flask(__name__,template_folder='templates')
model = pickle.load(open(modeloutput, "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)