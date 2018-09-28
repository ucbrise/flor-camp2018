# General Imports
import flor
import json
import pandas as pd
import scipy.sparse
import numpy as np
import time

# Imports for Pre-Processing
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

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


def preprocessing(data_loc, intermediate_X, intermediate_y, **kwargs):
    '''

    Data Preprocessing

    '''
    print("Data Preprocessing")
    data = pd.read_json(data_loc)
    X = data['text']
    y = data['rating'].astype(np.float64)
    
    en_stop = get_stop_words('en')

    def filter_sentence(el):
        tokens = word_tokenize(el)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = stem_words(tokens)
        tokens = lemma_words(tokens)

        ret_str = " ".join(tokens) 

        return ret_str 


    #Credit to https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
    #for stem_words and lemma_words
    def stem_words(words):
        stemmer = PorterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemma_words(words):
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    start_time = time.time()
    X = [filter_sentence(el) for el in X]
    print("--- %s seconds ---" % (time.time() - start_time))

    y_new = []
    for el in y:
        ret = 0
        if el <= 5:
            ret = 0
        else:
            ret = 1
        y_new.append(ret)
    y = y_new

    # Load the cleaned data
#     with open('data_clean_X.json') as json_data:
#         X = json.load(json_data)
#         json_data.close()
#     with open('data_clean_y.json') as json_data:
#         y = json.load(json_data)
#         json_data.close()

    with open(intermediate_X, 'w') as outfile:
       json.dump(X, outfile)
    with open(intermediate_y, 'w') as outfile:
       json.dump(y, outfile)
    


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
#     def train_drop(el):
#         tokens = word_tokenize(el)
#         tokens = [word for word in tokens if word.isalpha()]
#         ret_str = " ".join(tokens) 
#         return ret_str 
#     X_tr = [train_drop(el) for el in X_tr]
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

    

def train_test(X_train, X_test, y_train, y_test, report, **kwargs):
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
    clf = RandomForestClassifier(n_estimators=...).fit(X_train, y_train).fit(X_train, y_train)
#     clf = MultinomialNB().fit(X_train, y_train)
#     clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=hyperparameters, random_state=1).fit(X_train, y_train)\n",
#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hyperparameters, random_state=1).fit(X_train, y_train)
#     clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=hyperparameters, random_state=1, max_iter = 1000).fit(X_train, y_train)
#     clf = MultinomialNB().fit(X_train, y_train) #no hyperparameters needed
#     clf = KNeighborsClassifier().fit(X_train, y_train) #no hyperparameters needed

    print("Predicting Model")
    y_pred = clf.predict(X_test)
    
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
    
    
    