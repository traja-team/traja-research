import src.tf_src.models
from collections.abc import Iterable

def DeepLSTM(units: list, output_classes: int, backend='tensorflow'):
    """ Generates a deep LSTM with the provided width in each layer.

    Args:
        units (Iterable): Width of each layer
    """

    if backend == 'tensorflow':
        return src.tf_src.models.DeepLSTM(units=units, output_classes=output_classes)
    

# TODO figure out why Python cannot pick the correct function prototype
#def DeepLSTM(units: int, depth: int, output_classes: int, backend='tensorflow'):
#    """ Generates a deep LSTM with the provided width in each layer.
#
#    Args:
#        units (int): Width of each layer
#        depth (int): Number of layers
#    """
#    return DeepLSTM([units] * depth, output_classes, backend)
