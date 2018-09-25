import flor
import json
import pandas as pd
import numpy as np
import time

from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

@flor.func
def preprocessing(data_loc, intermediate_X, intermediate_y, **kwargs):
    '''

    Data Preprocessing

    '''
    print("Data Preprocessing")
#     data = pd.read_json(data_loc)
#     X = data['text']
#     y = data['rating'].astype(np.float64)
    
#     en_stop = get_stop_words('en')

#     def filter_sentence(el):
#         tokens = word_tokenize(el)
#         tokens = [word for word in tokens if word.isalpha()]
#         tokens = [word for word in tokens if word not in en_stop]
#         tokens = stem_words(tokens)
#         tokens = lemma_words(tokens)

#         ret_str = " ".join(tokens) 

#         return ret_str 


#     #Credit to https://www.kdnuggets.com/2018/03/text-data-preprocessing-walkthrough-python.html
#     #for stem_words and lemma_words
#     def stem_words(words):
#         stemmer = PorterStemmer()
#         stems = []
#         for word in words:
#             stem = stemmer.stem(word)
#             stems.append(stem)
#         return stems

#     def lemma_words(words):
#         lemmatizer = WordNetLemmatizer()
#         lemmas = []
#         for word in words:
#             lemma = lemmatizer.lemmatize(word, pos='v')
#             lemmas.append(lemma)
#         return lemmas

#     start_time = time.time()
#     X = [filter_sentence(el) for el in X]
#     print("--- %s seconds ---" % (time.time() - start_time))

#     y_new = []
#     for el in y:
#         ret = 0
#         if el <= 5:
#             ret = 0
#         else:
#             ret = 1
#         y_new.append(ret)
#     y = y_new

    # Load the cleaned data
    with open('data_clean_X.json') as json_data:
        X = json.load(json_data)
        json_data.close()
    with open('data_clean_y.json') as json_data:
        y = json.load(json_data)
        json_data.close()

    with open(intermediate_X, 'w') as outfile:
       json.dump(X, outfile)
    with open(intermediate_y, 'w') as outfile:
       json.dump(y, outfile)