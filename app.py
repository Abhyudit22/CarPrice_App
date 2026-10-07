# app.py - Car Price Prediction web application (Flask)
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model once, when the server starts
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        present_price = float(request.form["present_price"])
        kms_driven = int(request.form["kms_driven"])
        fuel_type = int(request.form["fuel_type"])        # Petrol 0, Diesel 1, CNG 2
        seller_type = int(request.form["seller_type"])    # Dealer 0, Individual 1
        transmission = int(request.form["transmission"])  # Manual 0, Automatic 1
        owner = int(request.form["owner"])
        car_age = int(request.form["car_age"])

        features = np.array([[present_price, kms_driven, fuel_type,
                              seller_type, transmission, owner, car_age]])
        price = round(float(model.predict(features)[0]), 2)
        return render_template("index.html", price=price, form=request.form)
    except (KeyError, ValueError):
        return render_template("index.html", error="Please check your input values.", form=request.form)


if __name__ == "__main__":
    app.run(debug=True)
