from flask import Flask, render_template, request
import os 
import pyrebase
from pyrebase.pyrebase import Storage 
import requests as req

config = {
    "apiKey": "AIzaSyAmFauSJcVZ6hDCHDe6R7u-udeFnDRsxgM",
    "authDomain": "agrobot-2477a.firebaseapp.com",
    "databaseURL": "https://agrobot-2477a-default-rtdb.firebaseio.com",
    "projectId": "agrobot-2477a",
    "storageBucket": "agrobot-2477a.appspot.com",
    "messagingSenderId": "498819222730",
    "appId": "1:498819222730:web:e5f8985474b9dbae463833",
    "measurementId": "G-4BSFR4WSL8"
}

firebase = pyrebase.initialize_app(config)

api_key = '827b9408e2582bbc7e4ec550027ccb62'

storage_ = firebase.storage()
db = firebase.database()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        name = request.form['name']
        db.child("todo").push(name)
        todo = db.child("todo").get()
        to = todo.val()
        return render_template('index.html', t=to.values())
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['img']
        lat = request.form['lat']
        lon = request.form['lon']
        #print(image.read())
        #!storage_.child('plants').put(image)
        url_weather = 'https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}'.format(lat, lon, api_key)
        res_weather = req.get(url_weather)
        return render_template('upload_image.html', image_up=res_weather.json())#res_weather)
    return render_template('upload_image.html')

#@app.route('/diagnosis', methods=['GET','POST'])
#def diagnosis():
#    if request.method == 'POST':
        
    #if request.method == 'POST':
    #    name = request.form
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
