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

class AutoEncoder(tf.keras.Model):
    def __init__(self,hidden_dims: int, latent_dims: int = 3, output_dims: int=2, **kwargs):
        """AutoEncoders with LSTM as hidden units and disentangled latent space

        Args:
            hidden_dims (list): Width of each hidden layer in encoder and decoder part of the network
            latent_dims (int, optional): Size of latent space. Defaults to 3.
            output_dims (int, optional): Number of features in the data. Defaults to 2.
        """
        super(AutoEncoder,self).__init__()
        self.hidden_dims = hidden_dims
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        
        self.encoder = LSTMEncoder(hidden_dims=hidden_dims)
        self.decoder = LSTMDecoder(hidden_dims=hidden_dims) 
        self.latent = Latent(latent_dims=latent_dims)
        self.out = OutputLayer(output_dims=output_dims)
        
        self.model = Sequential()
        self.model.add(self.encoder)
        self.model.add(self.latent)
        self.model.add(self.decoder)
        self.model.add(self.out)
        
    def get_config(self):
        config = super(AutoEncoder,self).get_config()
        return config
    
    def call(self,x):
        # x = self.encoder(x)
        # x = self.latent(x)
        # x = self.decoder(x)
        x = self.model(x)
        return x
    
class LSTMEncoder(tf.keras.Model):
        
    def __init__(self,hidden_dims):
        super(LSTMEncoder,self).__init__(self)
        self.hidden_layer = tf.keras.layers.LSTM(hidden_dims, return_sequences=True, return_state=True)
        
    def call(self,x):
        out,hidden_state, _= self.hidden_layer(x)
        return out
class Latent(tf.keras.Model):
    
    def __init__(self,latent_dims):
        super(Latent,self).__init__(self)
        self.hidden_layer = tf.keras.layers.Dense(latent_dims, activation='tanh')
        
    def call(self,x):
        out= self.hidden_layer(x)
        return out
class LSTMDecoder(tf.keras.Model):
    def __init__(self,hidden_dims):
        super(LSTMDecoder,self).__init__(self)
        self.hidden_layer = tf.keras.layers.LSTM(hidden_dims, return_sequences=True, return_state=True)
        
    def call(self,x):
        out,hidden_state,_= self.hidden_layer(x)
        return out 
class OutputLayer(tf.keras.Model):
    def __init__(self, output_dims):
        super(OutputLayer, self).__init__(self)
        self.out = tf.keras.layers.Dense(output_dims)
    def call(self, x):
        out = self.out(x)
        return out
    