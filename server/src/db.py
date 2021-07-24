import pyrebase 

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

db = firebase.database()

#db.child('names').push({"name":"luis"})
users = db.child("names").child("name").get()
print(users.val())