# flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
data = pd.read_csv("clean_data.csv")
pipe = pickle.load(open("Ridgemodel.pkl", 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    house_age = request.form.get('House Age')
    distance_from_metro = request.form.get('Distance From Metro')
    stores = request.form.get('No. of Stores')
    no_of_bedrooms = request.form.get('Number of bedrooms')
    house_size = request.form.get('House size (sqft)')
    price_per_unit = request.form.get('House price of unit area')
    input = pd.DataFrame([[house_age, distance_from_metro, stores, no_of_bedrooms, house_size, price_per_unit]],
                         columns=['House Age', 'Distance from nearest Metro station (km)',
                                  'Number of convenience stores', 'Number of bedrooms', 'House size (sqft)',
                                  'House price of unit area'])
    prediction = pipe.predict(input)[0]
    return str(np.round(prediction,2))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
