# from flask import Flask,request,render_template
# app = Flask(__name__,template_folder='template')
# app.debug = True

# @app.route("/")
# def hello_world():
#     return render_template("index.html")

# @app.route('/sample_registration')
# def predict():
#     return render_template("sample_registration.html")

# if __name__== "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template, send_from_directory
from flask.templating import render_template
from flask_sqlalchemy import SQLAlchemy
import os
import json
import numpy as np
import pickle

# Import for Migrations
from flask_migrate import Migrate, migrate

# Load data from JSON file
with open(os.path.abspath('static/data.json')) as f:
    area_data = json.load(f)

app = Flask(__name__, template_folder='template')
app.debug = True

# adding configuration for using a sqlite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'

# Creating an SQLAlchemy instance
db = SQLAlchemy(app)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Models


class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    state = db.Column(db.String(100), unique=False, nullable=False)
    district = db.Column(db.String(100), unique=False, nullable=False)
    block = db.Column(db.String(100), unique=False, nullable=False)
    village = db.Column(db.String(100), unique=False, nullable=False)
    name = db.Column(db.String(100), unique=False, nullable=False)
    phone = db.Column(db.Integer, nullable=False)
    prev_crop = db.Column(db.String(100), unique=False, nullable=False)

    crop_group = db.Column(db.String(100), unique=False, nullable=True)
    crop = db.Column(db.String(100), unique=False, nullable=True)
    variety = db.Column(db.String(100), unique=False, nullable=True)
    season = db.Column(db.String(100), unique=False, nullable=True)
    soil_type = db.Column(db.String(100), unique=False, nullable=True)
    crop_duration = db.Column(db.String(100), unique=False, nullable=True)
    irrigation = db.Column(db.String(100), unique=False, nullable=True)

    # repr method represents how one object of this datatable
    # will look like
    def __repr__(self):
        return f"Name : {self.name}"


migrate = Migrate(app, db, render_as_batch=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/static/<path:path>')
def serve_static(path):
    """ Serve static files from the 'static' directory """
    return send_from_directory('static', path)


@app.route('/pages/<file_name>', methods=["GET", "POST"])
def pages(file_name):
    return render_template(file_name)


@app.route('/get_area_data/<filter>', methods=["GET"])
def get_area_data(filter):
    if filter == "state":
        return json.dumps({"states": list(area_data.keys())})
    elif filter == "district":
        state = request.args.get("state")
        return json.dumps({"districts": list(area_data[state].keys())})
    elif filter == "block":
        state = request.args.get("state")
        district = request.args.get("district")
        return json.dumps({"blocks": list(area_data[state][district].keys())})
    elif filter == "village":
        state = request.args.get("state")
        district = request.args.get("district")
        block = request.args.get("block")
        return json.dumps({"villages": list(area_data[state][district][block])})


@app.route('/pages/<crop_details>', methods=["GET", "POST"])
def index2():
    if request.method == "POST":
        pid = request.form.get("id")
        data = Profile.query.get(pid)

        data.crop_group = request.form.get("crop_group")
        data.crop = request.form.get("crop")
        data.season = request.form.get("season")
        data.soil_type = request.form.get("soil_type")
        data.crop_duration = request.form.get("crop_duration")
        data.irrigation = request.form.get("irrigation")

        db.session.commit()
        return render_template("crop_details.html", id=pid, data=",".join([str(data.crop), str(data.variety)]))
    else:
        return render_template("sample_registration.html")


@app.route('/predict', methods=["GET", "POST"])
def predict():

    xrf_fe = float(request.form.get("xrf_fe"))
    xrf_k = float(request.form.get("xrf_k"))
    xrf_ti = float(request.form.get("xrf_ti"))
    xrf_ca = float(request.form.get("xrf_ca"))
    xrf_ba = float(request.form.get("xrf_ba"))
    xrf_zr = float(request.form.get("xrf_zr"))
    xrf_mn = float(request.form.get("xrf_mn"))
    xrf_co = float(request.form.get("xrf_co"))
    xrf_cr = float(request.form.get("xrf_cr"))
    xrf_v =  float(request.form.get("xrf_v"))
    xrf_sr = float(request.form.get("xrf_sr"))
    xrf_zn = float(request.form.get("xrf_zn"))
    xrf_sb = float(request.form.get("xrf_sb"))
    xrf_pb = float(request.form.get("xrf_pb"))
    xrf_ag = float(request.form.get("xrf_ag"))
    xrf_as = float(request.form.get("xrf_as"))

    full_inp = (xrf_fe, xrf_k, xrf_ti, xrf_ca, xrf_ba, xrf_zr, xrf_mn, xrf_co, xrf_cr, xrf_v, xrf_sr, xrf_zn, xrf_sb,xrf_pb, xrf_ag, xrf_as)
    
    def func_agc(full_inp):
        with open('static/models/agc_xgb.pkl', 'rb') as f:
            xgb = pickle.load(f)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(full_inp)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = xgb.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            print('The agro-climatic region is Coastal Saline zone')
        elif (prediction[0]==1):
            print('The agro-climatic region is Gangetic Alluvial Zone')
        elif (prediction[0]==2):
            print('The agro-climatic region is Red and Lateritic zone ')
        elif (prediction[0]==3):
            print('The agro-climatic region is Terai-Teesta Alluvial ')
        elif (prediction[0]==4):
            print('The agro-climatic region is Vindhyachal alluvial zone')
        else:
            print('The agro-climatic region is northern hilley zone ')  

        return prediction 
    
    def func_pm(full_inp):
        with open('static/models/pm_lgb.pkl', 'rb') as f:
            lgb = pickle.load(f)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(full_inp)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = lgb.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            print('The parent material is Recent alluvium')
        elif (prediction[0]==1):
            print('The parent material is old alluvium')
        elif (prediction[0]==2):
            print('The parent material is Granite-Gneiss')
        else:
            print('The parent material is deltaic alluvium')

        return prediction
    
    def func_so(full_inp):
        with open('static/models/so_xgb.pkl', 'rb') as f:
            xgb = pickle.load(f)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(full_inp)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = xgb.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            print('Soil Order is alfisol')
        elif (prediction[0]==1):
            print('The Soil Order is inceptisol')
        else:
            print('The Soil Order is entisol')
        return prediction
    
    def func_pp(full_inp):
        with open('static/models/pp_xgb.pkl', 'rb') as f:
            xgb = pickle.load(f)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(full_inp)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = xgb.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            print('Soil productivity potential high')
        elif (prediction[0]==1):
            print('Soil productivity potential medium')
        else:
            print('Soil productivity potential low')
        return prediction


    def func_OC(full_inp):
        with open('static/models/OC_rfc.pkl', 'rb') as f:
            rfc = pickle.load(f)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(full_inp)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = rfc.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            print('Organic Carbon in soil medium')
        elif (prediction[0]==1):
            print('Organic Carbon in soil low')
        else:
            print('Organic Carbon in soil high')
        return prediction
    
    def func_pH(full_inp):
        with open('static/models/pH_rfc.pkl', 'rb') as f:
            rfc = pickle.load(f)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(full_inp)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = rfc.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
            print('Soil is moderately to slightly acidic')
        elif (prediction[0]==1):
            print('Soil is neutral to slightly alkaline')
        else:
            print('Soil is strongly to highly acidic')
        return prediction
    
    agc_result = func_agc(full_inp)
    pm_result = func_pm(full_inp)
    so_result = func_so(full_inp)
    pp_result = func_pp(full_inp)
    OC_result = func_OC(full_inp)
    pH_result = func_pH(full_inp)

    result = {"result":[agc_result,pm_result,so_result,pp_result,OC_result,pH_result]}
    return render_template("result.html")
    
    
    
   

if __name__ == "__main__":
    app.run(debug=True)
