from tf_notebook import saturation_logger, target_layer
import unittest
from delve_utils import get_history, SimpsonDiversityIndexBasedSaturation, get_transformed_eig, get_projected_points
import numpy as np 

class TestTraja(unittest.TestCase):
    
    def test_saturation_logger(self):
        # Check saturation logger is not empty dict
        
            
    def test_get_history(saturation_logger,target_layer):
        
        history = get_history(saturation_logger=saturation_logger, target_layer=target_layer)
        pass
        