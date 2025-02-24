from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from sortedcontainers import SortedDict

class CGM:
    # in the future, want to be able to configure 
    def __init__(self, config={}):
        self.t = 0
        self.history: Dict[int, float]= SortedDict()
     
    def get_reading(self, t):
        return int(self.history[t])
                
    def post_reading(self, bg, t):
        self.history[t] = bg
        
class ErrorCGM(CGM):
    def get_reading(self, t):
        real = super().get_reading(t)
        
        
class BasalAttackCGM(CGM):
    def get_reading(self, t):
        real = super().get_reading(t)
        if t <= 15:
            return real
        if real <= 90:
            return 81
        return real
        