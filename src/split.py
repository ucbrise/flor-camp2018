import flor
import json
import scipy.sparse
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
#     def train_drop(el):
#         tokens = word_tokenize(el)
#         tokens = [el for el in tokens if random.random() >= 0.75]
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