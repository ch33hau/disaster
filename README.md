# Disaster Response Pipeline Project

## Installation

The code is expected to run in `python3`, with below dependencies installed:
- json
- plotly
- pandas
- nltk
- flask
- sklearn
- sqlalchemy
- sys
- numpy
- re
- pickle
- warnings

## Project overview

This project uses machine learning to convert disaster message into relevant categories so that it could be used in emergency situation.

It also includes the steps of cleaning data, and training for the model.

All the outcomes of above will be then converted to a python Flask app, which allows user to test the features on their browser.

## File Descriptions

- `app`: This folder contains all python code:
  - `run.py`: The code for the web app
  - `process_data.py`: The code for running ETL pipeline and cleans data
  - `train_classifier.py`: The code for running ML pipeline that trains classifier and create the ML model
- `data`: This folder contains the original messages and categories data, and SQLite file will be stored here as well
- `models`: This folder uses to store ML model
- `templates`: The template folder for Flask app
- `ETL Pipeline Preparation.ipynb`: The exploration notebook that helps to develop `process_data.py` for ETL pipeline.
- `ML Pipeline Preparation.ipynb`: The exploration notebook that helps to develop `train_classifier` for ML pipeline.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 app/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python3 app/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 app/run.py`

3. Go to http://0.0.0.0:3001/
