from constants import *
import pandas as pd
import os

def _read_file(path):
    with open(path) as f:
        o = f.read().strip()
    return o

def _parse_path(path):
    base = os.path.basename(path)
    i, rating = base.split('.')[0].split('_')
    return rating


def load_data():
    df = pd.DataFrame(columns=('text', 'rating'))
    i = 0
    for root in ALL_PATHS:
        for datum in os.listdir(root):
            path = os.path.join(root, datum)
            df.loc[i] = [_read_file(path), _parse_path(path)]
            i += 1
            print(i)
    return df

