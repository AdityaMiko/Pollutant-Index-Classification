import numpy as np 
from flask import Flask, request, render_template
import pickle

#creat Flask App
app = Flask(__name__)

#import model klasifikasi
model = pickle.load(open("model.pkl","rb"))

#route home
@app.route("/")
def Home():
    return render_template("index.html")

#route prediksi
@app.route("/predict",methods = ["POST"])
def predict():
    float_features = [float(x)for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    result_classification = ', '.join(prediction)

    return render_template("index.html",prediction_text = "The polutant index is {}".format(result_classification))

if __name__ == "_main_":
    app.run()