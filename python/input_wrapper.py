#import sys
#sys.path.append('/home/vbhandare/.local/lib/python2.7/site-packages')
import numpy as np

class Input :
    def __init__(self) :
        self.input_params = ['age', 
                             'wbc', 
                             #'neutrophil count', 
                             #'lymphocyte count', 
                             'nlr', 
                             'ast', 
                             'albumin', 
                             'ldh', 
                             'crp'
                             ]
        self.input_values = []
        self.param_length = len(self.input_params)
        self.estimator_id = None

    def add_param(self, param) :
        self.input_params.append(param)

    def add_value(self, value) :
        self.input_values.append(value)

    def get_all_params(self) :
        return self.input_params
    
    def get_all_values(self) :
        return self.input_values
    
    def get_value(self, index) :
        return self.input_values[index]
        
    def get_param(self, index) :
        return self.input_params[index]
    
    def get_ndarray(self) :
        return np.array(self.input_values).reshape(1, -1)
