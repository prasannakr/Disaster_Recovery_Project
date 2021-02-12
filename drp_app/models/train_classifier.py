import sys
import re
import joblib 
import numpy as np
import pandas as pd
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from workspace_utils import active_session


def load_data(database_filepath = 'DisasterResponse.db', table_name = 'Messages', column_name='message'):
    '''
    Load database and get the data
    Provide path to database/table name
    Returns X (features) Y (target variables)
    '''
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, con = engine)
    X = df[column_name]
    Y = df[df.columns[5:]]
    
    return X, Y, Y.columns
    

def tokenize(text):
    '''
    This function accepts text message as input.
    Then normalize the text, remove punctuation, tokenize, remove stop words, reduce words to their stem & then to their root form
    '''

    #Normalize text & remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    #Tokenize text, remove stop words
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
        
    # Reduce words to their stems
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    
    return lemmed    

def build_model():
    '''
    This machine pipeline should take in the message column as input and output classification results on the other 36 categories in the dataset
    '''
    pipeline = Pipeline([('vectorizer', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced')))])
    
    
    parameters = {
                'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
                'tfidf__use_idf': (True, False)
                 }
    
    gs_clf = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    return gs_clf
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Predict values based on the model &
    Print classification report
    '''
    
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath='onevsrest_linear_best.pkl'):
    '''
    saves the model to the model filepath
    '''
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
