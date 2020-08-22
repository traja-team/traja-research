import tensorflow as tf

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, LSTM, RepeatVector, Dropout

from collections.abc import Iterable

def DeepLSTM(units: Iterable=[512,512], output_classes: int=2):
    """ Generates a deep LSTM with the provided width in each layer.

    Args:
        units (Iterable): Width of each layer
    """
    
    model = Sequential()
    for unit in units:
        model.add(LSTM(units=unit, return_sequences=True))
    model.add(Dense(output_classes, activation='sigmoid'))

    return model
