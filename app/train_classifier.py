import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')


def load_data(database_filepath):
    """
    Descriptions:
    Read data from file
    
    Arguments:
    database_filename: string. File name to the data
       
    Returns:
    features: dataframe. Features dataset.
    labels: dataframe. Labels dataset.
    category_names: strings. Category names.
    """
    # Load data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    
    features = df['message']
    labels = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(labels.columns.values)
    
    return features, labels, category_names


def tokenize(text):
    """
    Description:
    Tokenize the input text
    
    Arguments:
    text: string. Message string
       
    Returns:
    stemmed: strings. Stemmed word tokens
    """
    
    # Remove symbols
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize message string
    tokens = word_tokenize(text)
    
    # Cleanup stop words and apply stem
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")

    return [stemmer.stem(word) for word in tokens if word not in stop_words]


def get_scorer(y_true, y_pred):
    """
    Description:
    Calculate median F1 score for all of the output classifiers
        
    Arguments:
    y_true: array. Actual labels
    y_pred: array. Predicted labels.
    
    Returns:
    score: float. Median F1 score for all of the output classifiers
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_list.append(f1)
        
    score = np.median(f1_list)
    return score


def build_model():
    """
    Description:
    Build a machine learning pipeline
    
    Arguments:
    None
       
    Returns:
    cv: gridsearchcv object. Gridsearchcv object that trained and figured out with optimal parameters
    """
    MIN_DF = 5
    N_ESTIMATORS = 10
    MIN_SAMPLES_SPLIT = 10
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, min_df = MIN_DF)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = N_ESTIMATORS,
                                                             min_samples_split = MIN_SAMPLES_SPLIT)))
    ])
    
    # Define possible parameters
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[10, 25], 
                  'clf__estimator__min_samples_split':[2, 5, 10]}
    
    # Create scorer
    scorer = make_scorer(get_scorer)
    
    return GridSearchCV(pipeline, param_grid = parameters, scoring = scorer, verbose = 10)


def evaluate_metrics(actual, predicted, col_names):
    """Evaluate metrics for ML model
    
    Args:
    actual: array. Actual labels.
    predicted: array. Predicted labels.
    col_names: strings. Category names.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall and f1 score.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description:
    Evaluate test accuracy, precision, recall and F1 score for fitted model
    
    Arguments:
    model: model object. Fitted model object.
    X_test: dataframe. Test features dataset.
    Y_test: dataframe. Test labels dataset.
    category_names: strings. Category names.
    
    Returns:
    None
    """
    # Predict from input features
    Y_pred = model.predict(X_test)
    
    # Evaluate and print result
    eval_metrics = evaluate_metrics(np.array(Y_test), Y_pred, category_names)
    print(eval_metrics)


def save_model(model, model_filepath):
    """
    Description:
    Save model to file
    
    Args:
    model: model object. Fitted model object.
    model_filepath: string. Target save path
    
    Returns:
    None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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