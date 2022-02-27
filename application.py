from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("LinearRegressionModel.pkl",'rb'))
car = pd.read_csv('cardekho_new_updated.csv')

@app.route('/')
def index():
    companies = sorted(car['full_name'].str.split(' ').str.slice(0,1).str.join(' ').unique())
    car_models = sorted(car['full_name'].str.split(' ').str.slice(0,3).str.join(' ').unique())
    year = sorted(car['year'].unique(), reverse=True)
    seller_type = sorted(car['seller_type'].unique())
    owner_type = sorted(car['owner_type'].unique())
    fuel_type = sorted(car['fuel_type'].unique())
    transmission_type = sorted(car['transmission_type'].unique())
    seats = [5,7]
    return render_template('index.html', companies=companies, car_models=car_models, years=year, seller_types=seller_type, owner_types=owner_type , fuel_types=fuel_type, transmission_types=transmission_type, seats=seats)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    seller_type = request.form.get('seller_type')
    owner_type = request.form.get('owner_type')
    fuel_type = request.form.get('fuel_type')
    transmission_type = request.form.get('transmission_type')
    seats = float(request.form.get('seats'))
    km_driven = int(request.form.get('kilo_driven'))
    mileage = float(request.form.get('mileage'))
    engine = float(request.form.get('engine'))
    max_power = float(request.form.get('max_power'))


    prediction = model.predict(pd.DataFrame([[car_model, year, seller_type, km_driven, owner_type, fuel_type, transmission_type, mileage, engine, max_power, seats]], columns=['full_name','year','seller_type','km_driven','owner_type','fuel_type','transmission_type','mileage','engine','max_power','seats'] ))

    if prediction[0]<0:
        prediction[0] = 1000

    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)