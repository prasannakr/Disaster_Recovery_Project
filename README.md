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
5) Steps used to deploy in Heroku<br/>
a) Create a directory: mkdir disaster_response_app<br/>
b) Move files to above directory: mv -t disaster_response_app app data models DisasterResponse.db <br/>
c) Updata python(if not latest version): conda install python<br/>
d) Create Virtual Environment: python3.5 -m venv disastervenv<br/>
e) Activate above: source disastervenv/bin/activate<br/>
f) Move to the directory: cd disaster_response_app<br/>
g) Install required packages: pip install flask pandas plotly gunicorn<br/>
h) Get Heroku CLI: curl https://cli-assets.heroku.com/install-ubuntu.sh | sh<br/>
i) Login to Heroku: heroku login -i<br/>
j) Comment out app.run() in run.py file<br/>
k) Create Procfile: touch Procfile<br/>
l) Open Procfile and type: web gunicorn run:app<br/>
m) Create requirements file: pip3 freeze > requirements.txt<br/>
n) Git commands:<br/>
i> git init<br/>
ii> heroku git:remote -a krp-disaster-response-app<br/>
iii> git add .<br/>
iv> git commit -am "make it better"<br/>
v) git remote -v<br/>
vi> git push heroku master<br/>
o) Link to access app - https://krp-disaster-response-app.herokuapp.com/<br/>


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

