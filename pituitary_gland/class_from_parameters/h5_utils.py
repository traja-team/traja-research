import os
import pandas as pd


EXPERIMENTS_DIR = '.'
DATASETS_FILE = 'datasets.h5'


def list_keys():
    with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
        print(list(store.keys()))


def delete_key(key: str):
    with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
        del store[key]


def delete_keys(keys: list):
    for key in keys:
        delete_key(key)


def clear_experiments_in_progress():
    with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
        for key in store.keys():
            if key.endswith("currently_running"):
                delete_key(key)

list_keys()
