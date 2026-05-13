from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

#route for Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            gender = request.form['gender']
            race_ethnicity = request.form['race_ethnicity']
            parental_level_of_education = request.form['parental_level_of_education']
            lunch = request.form['lunch']
            test_preparation_course = request.form['test_preparation_course']

            reading_score = int(request.form['reading_score'])
            writing_score = int(request.form['writing_score'])

            data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score,
            )

            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=round(results, 2))

        except Exception as e:
            print("ERROR:", e)
            return render_template('home.html', results="Invalid Input")


if __name__ == '__main__':
    app.run(host='0.0.0.0')









