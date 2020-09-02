import tensorflow as tf 
from tensorflow import keras 
import numpy as np  
from numpy.linalg import LinAlgError

def get_history(saturation_logger, target_layer):
    """Returns array of history of activations of the target layer"""
    layer_history = saturation_logger.preactivation_states[target_layer]
    history = np.stack(layer_history)[:, 0, :] 
    flattened_shape = (history.shape[0], history.shape[1] * history.shape[2])
    history = np.reshape(history, flattened_shape)  
    return history
  
def get_preactivation_tensors(layers):
    """Get all valid layers for computing saturation."""
    dense_outputs = []
    for layer in layers:
        if 'dense' in layer.name or 'lstm' in layer.name:
            if '_input' in layer.name:
                # HACK
                continue
            # Get pre-activation
            if hasattr(layer, 'activation') and layer.activation.__name__ != \
                    'linear':
                preactivation_tensor = layer.output.op.inputs[0]
            else:
                preactivation_tensor = layer.output
            dense_outputs.append(preactivation_tensor)
    return dense_outputs

def initialize_preactivation_states(dense_outputs, obj):
    """Creates lists for `preactivation_states` dictionary."""
    for tensor in dense_outputs:
        layer_name = tensor.name.split('/')[0]
        obj.preactivation_states[layer_name] = []

def get_layer_outputs(obj):
    """Get intermediate outputs aka. preactivation states."""
    layers = obj.model.layers[1:]
    dense_outputs = get_preactivation_tensors(layers)
    return dense_outputs


def save_intermediate_outputs(dense_outputs, obj):
    """Save outputs to obj."""
    for tensor in dense_outputs:
        layer_name = tensor.name.split('/')[0]

        # Route intermediate output, aka. preactivation state
        #print("Input", obj.model.input)
        #print("Phase", keras.backend.learning_phase())
        #print("Add", [obj.model.input] + [keras.backend.learning_phase()])
        #print("tensor", tensor)
        #print("List tensor", [tensor])
        #func = keras.backend.function([obj.model.input] + [keras.backend.learning_phase()], [tensor])
        func = keras.backend.function([obj.model.input], [tensor])
        #intermediate_output = func([obj.input_data, 0.])[0]  # batch_nr x width
        #print("Input data", obj.input_data)
        intermediate_output = func(obj.input_data)[0] 

        obj.preactivation_states[layer_name].append(intermediate_output)

class SaturationMetric(keras.callbacks.Callback):
    """Keras callback for computing and logging layer saturation.
        Args:
            model: Keras model
            input_data: sample input to calculate layer saturation with, eg train
            print_freq
    """
    def __init__(self, model, input_data, print_freq=1):
        self.model = model
        self.input_data = input_data
        self.print_freq = print_freq

    def on_train_begin(self, logs=None):
        self.preactivation_states = {}
        layers = self.model.layers
        dense_outputs = get_preactivation_tensors(layers)
        initialize_preactivation_states(dense_outputs, self)

    def on_batch_end(self, batch, logs):
        if batch % 10 == 0:
            # TODO Check if has activation
            dense_outputs = get_layer_outputs(self)
            save_intermediate_outputs(dense_outputs, self)

    def on_epoch_end(self, epoch, logs):
        layers = self.preactivation_states.keys()
        logs = record_saturation(layers, self, epoch, logs, self.model)
        if epoch > 2:
            for layer in layers:
                try:
                    print("epoch = %4d  layer = %r  sat = %0.2f%%" \
                          % (epoch, layer, logs[layer]))
                    logs[layer] = self.preactivation_states[layer]
                except Exception as e:
                    print(e)

def get_projected_points(transformation_matrix,history,layer,epoch):
    """ """
    projected_points = dict()
    if layer not in projected_points:
        projected_points[layer] = list()
        if len(projected_points[layer]) <= epoch:
            projected_points[layer].append(list())

        for index in range(history.shape[0]):
            projected_output = np.matmul(transformation_matrix, history[index]) 
            projected_points[layer][epoch].append(projected_output[0:2])
            
    return projected_points

def record_saturation(layers: str,
                      obj,
                      epoch: int,
                      logs: dict,
                      model,
                      write_summary: bool = False):
    """Records saturation for layers into logs and writes summaries."""
    for layer in layers:
        layer_history = obj.preactivation_states[layer]
        if len(layer_history) < 2:  # ?
            continue
        history = np.stack(
            layer_history)[:, 0, :]  # get first representation of each batch

        # TODO check if we should flatten by multiplying
        # the time series entries with the output units
        flattened_shape = (history.shape[0], history.shape[1] * history.shape[2])
        history = np.reshape(history, flattened_shape)
        
        history_T = history.T
        try:
            cov = np.cov(history_T)
        except LinAlgError:
            continue
        eig_vals, eig_vecs = np.linalg.eigh(cov)

        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                     for i in range(len(eig_vals))]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = sorted(eig_pairs, key=lambda x: x[0], reverse=True)
        eig_vals, eig_vecs = zip(*eig_pairs)
        tot = sum(eig_vals)

        # Get explained variance
        var_exp = [(i / tot) for i in eig_vals]

        # Get Simpson-diversity-index-based saturation
        weighted_sum = sum([x**2 for x in var_exp])  #
        logs[layer] = weighted_sum
        if write_summary:
            tf.summary.scalar(layer,
                              weighted_sum,
                              collections=['preactivation_state'])        
    return logs

def get_transformed_eig(eig_pairs):
    eigen_space = eig_pairs[0:2]
    eigen_space = np.array([eig_pairs[0][1], eig_pairs[1][1]])
    transformation_matrix = np.matmul(np.transpose(eigen_space), eigen_space)
    return transformation_matrix


def SimpsonDiversityIndexBasedSaturation(history):
    
    """
    Compute the covariance of preactivation history, sorted list of eigen pairs(values,vectors), 
    and their variance(explained) followed by their weighted sum
    
    Returns: eigen pairs and weighted sum of explained variance"""
    
    cov = np.cov(history.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                        for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda x: x[0], reverse=True)
    eig_vals, eig_vecs = zip(*eig_pairs)
    tot = sum(eig_vals)
    # Get explained variance
    var_exp = [(i / tot) for i in eig_vals]

    # Get Simpson-diversity-index-based saturation
    weighted_sum = sum([x**2 for x in var_exp]) 
    
    return eig_pairs, weighted_sum     


