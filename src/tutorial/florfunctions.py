# General Imports
import flor
import json
import pandas as pd
import scipy.sparse
import numpy as np
import time

# Imports for train/test Split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Imports for Training and Testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import average_precision_score, recall_score, precision_recall_fscore_support
    

@flor.func
def traintest_split(intermediate_X, intermediate_y, X_train, X_test, y_train, y_test, **kwargs):
    '''

    Flor function to perform train/test split.

    '''
    with open(intermediate_X) as json_data:
        X = json.load(json_data)
        json_data.close()
    with open(intermediate_y) as json_data:
        y = json.load(json_data)
        json_data.close()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=92)

    vectorizer = TfidfVectorizer()
    start_time = time.time()
    vectorizer.fit(X_tr)
    X_tr = vectorizer.transform(X_tr)
    X_te = vectorizer.transform(X_te)
    with open(y_train, 'w') as outfile:
        json.dump(y_tr, outfile)
    with open(y_test, 'w') as outfile:
        json.dump(y_te, outfile)

    print("saving sparse matrices")
    scipy.sparse.save_npz(X_train, X_tr)
    scipy.sparse.save_npz(X_test, X_te)

    
@flor.func
def train_test(X_train, X_test, y_train, y_test, hyperparameters, report, **kwargs):
    '''

    Flor function to train and test with hyperparameters.

    '''
    print("Loading Data")
    X_train = scipy.sparse.load_npz(X_train)
    X_test = scipy.sparse.load_npz(X_test)
    with open(y_train) as json_data:
        y_train = json.load(json_data)
        json_data.close()
    with open(y_test) as json_data:
        y_test = json.load(json_data)
        json_data.close()
    print("Training Model")
    
    #Either train Random Forest or Multi-layer Perception Classifier
    clf = RandomForestClassifier(n_estimators=hyperparameters).fit(X_train, y_train).fit(X_train, y_train)
#     clf = MultinomialNB().fit(X_train, y_train)

    print("Predicting Model")
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print("Writing Results") 
    
    #fix this to output dataframe
    c = classification_report(y_test, y_pred).split('\n')
    
    output = []
    for x in range(len(c)):
        if x>2 and c[x]!='':
            temp = {}
            data = []
            for each in c[x].split('    '):
                if each != '':
                    data.append(each)
            temp['class'] = data[0]
            temp['precision'] = float(data[1])
            temp['recall'] = float(data[2])
            temp['f1-score'] = float(data[3])
            temp['support'] = float(data[4])
            output.append(temp)
            
    pd.DataFrame.from_dict(output).to_csv(report)
    
    return {'score': score}
    