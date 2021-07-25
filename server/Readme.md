# Biobot Backend Package

Install backend biobot package before run the rest api

First install the requirements

```bash
pip install -r requirements.txt
```

Install the package biobot

```bash
cd server
pip install -e .
```

Run server REST API with the two core-services, please copy the credentials in a textfile in this level called -->  ".openaikey.txt". Ensure to create this on the server-side as this file is in the .gitignore.

```bash
python3 rest_app,oy
```

## DONE LIST

* backend as a module
* Load the model just once
* Rest API definition for the both services.

## TODO LIST

* Missing Authentification with IBM
* Download the model from IBM Storage Service
* Prepare the suggested question after prediction - before start chatbot.
* move the deployment files (e.g. dockerfiles manifests) inside the server folder if you are planning to upload in this repo the frontend code. But I suggest to have annother one.
