import flor
import json
import scipy.sparse

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
def train_test(X_train, X_test, y_train, y_test, hyperparameters, precision, recall, **kwargs):
    '''

    Flor function to train and evaluate model.

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
    #FIXME! A model is missing!

    print("Predicting Model")
    y_pred = clf.predict(X_test)
    
    print("Writing Results") 
    
    #fix this to output dataframe
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    
    #change to output dataframe
    #Write the precision to the output file
    output = open(precision, 'w')
    output.write(str(hyperparameters) + '\n')
    output.write(str(prec))
    output.close()
    
    #Write the recall to the output file
    output = open(recall, 'w')
    output.write(str(hyperparameters) + '\n')
    output.write(str(rec))
    output.close()