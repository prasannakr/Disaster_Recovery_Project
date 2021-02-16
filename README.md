# Disaster Response Pipeline Project

## **Table of Contents:**
1. [Project Introduction](README.md#project-introduction)
2. [File Description](README.md#file-description)
3. [Instructions](README.md#Instructions)
4. [Libraries used](README.md#libraries-used)
5. [Results](README.md#results)
6. [Licensing, Acknowledgements](README.md#licensing-acknowledgements)

## **Project Introduction**<br/>

In this project, I will analyzing data provided by [Figure Eight](https://appen.com/). <br/>
Data contains pre-labeled tweets and text messages that are received during real life disasters. <br/>
Objective is to prepare the data with ETL(Extract, Transform, Load) pipeline & then use a ML(Machine Learning) pipeline to build a supervised learning model to categorize the events and look out for any trends.<br/> 
This will help emergency workers to classify/categorize the events and send the messages to appropriate disaster relief agency.

## **File Description**<br/>

There are two jupyter notebook files which contain code executed successfully:<br/>
1) ETL Pipeline preparation > Contains code for extracting/cleaning/wrangling/loading final data into sqlite database.<br/>
2) ML Pipeline preparation > Contains code for modeling, using pipeline, gridsearch & few models were run & model with better f1 score chosen.<br/>

There are three python scripts used to deploy on the workspace:<br/>
3) process_data > Contains functions with executed code from ETL Pipeline<br/>
4) train_classifer > contains functions with executed code from ML Pipeline<br/>
5) run > contains model pkl file and code for visualizations to run the web app<br/>

## **Instructions**<br/>

1) To run ETL pipeline that cleans data and stores in database:<br/>
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db<br/>
2) To run ML pipeline that trains classifier and saves<br/>
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl<br/>
3) To run web app in the app's directory<br/>
python run.py<br/>
4) URL to see visualization<br/>
https://view6914b2f4-3001.udacity-student-workspaces.com/<br/>


## **Libraries Used**<br/>

Following libraries were used:<br/>
Plotly<br/>
joblib<br/>
Pandas<br/>
Numpy<br/>
nltk<br/>
flask<br/>
sqlalchemy<br/>
sys<br/>
scikit-learn<br/>

## **Results**<br/>
End result is a web app powered by supervised machine learning model (LinearSVC) which:<br/>
A) Accepts a text message<br/>
B) Categorizes text message into appropriate groups<br/>
C) Few visualizations(bar charts) like Distribution of Message Genres/Distribution of Message Categories/Top words<br/>

## **Licensing, Acknowledgements**<br/>
Thanks to real life disaster messages data from Figure Eight.<br/>
Thanks to Udacity for providing knowledge on Data Engineering (ETL/NLP/ML Pipelines) and a platform to work on this project.<br/>

