from numpy.linalg.linalg import eig
import src.models
import src.tf_src.datasets
import src.tf_src.callbacks
from src.tf_src.metrics import SaturationLogger
from src.tf_src.loss import loss_mse_warmup
from delve_utils import get_history, SimpsonDiversityIndexBasedSaturation, get_transformed_eig, get_projected_points

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.optimizers import RMSprop, Adam


src.models.DeepLSTM(units=[512,512,512], output_classes=2)
src.models.DeepLSTM(100, 3, 2)
src.tf_src.datasets.E1_E2()

def setup():

    model = src.models.DeepLSTM(units=[512,512,512], output_classes=2)

    generator, data = src.tf_src.datasets.E1_E2()

    x_train, x_test, y_train, y_test = data

    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    
    validation_data = (np.expand_dims(x_test_scaled, axis=0),
                       np.expand_dims(y_test_scaled, axis=0))

    x_batch, y_batch = next(generator)
    saturation_logger = SaturationLogger(model, input_data=x_batch[:2], print_freq=1)

    callbacks = [src.tf_src.callbacks.callback_early_stopping,
                 src.tf_src.callbacks.callback_checkpoint,
                 src.tf_src.callbacks.callback_tensorboard,
                 src.tf_src.callbacks.callback_reduce_lr,
                 saturation_logger]

    return model, generator, validation_data, callbacks


if __name__ == '__main__':
    model, generator, validation_data, callbacks = setup()
    optimizer = RMSprop(lr=1e-3, momentum=0.0)
    model.compile(loss=loss_mse_warmup, optimizer=optimizer)
    model.fit(x=generator,
              epochs=1, #40,
              steps_per_epoch=1, #100,
              validation_data=validation_data,
              callbacks=callbacks)
    
    '''Get the embeddings of target layer after training'''
    
    # Instance of Saturation logger borrowed from delve
    saturation_logger = SaturationLogger(model, input_data=validation_data[:2], print_freq=1)  # Replaced x_data --> validation data
    target_layer ='lstm_4'
    projected_points = dict()
    
    history = get_history(saturation_logger=saturation_logger, target_layer=target_layer)
    eig_pairs, weighted_sum = SimpsonDiversityIndexBasedSaturation(history) 
    transformation_matrix = get_transformed_eig(eig_pairs=eig_pairs)
    projected_points = get_projected_points(transformation_matrix=transformation_matrix,history=history,layer=target_layer,epoch=1)
    print(projected_points.keys())
    
    
    
    