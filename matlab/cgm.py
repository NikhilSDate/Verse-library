from typing import Dict, List, Tuple
import numpy as np

class CGM:
    # in the future, want to be able to configure 
    def __init__(self, config={}):
        self.t = 0
        self.history: Dict[int, float]= {}
    
    def get_reading(self, t):
        # time = max(filter(lambda s: s % 5 == 0 and s <= t, self.history.keys()))
        return self.history[t]
                
    def post_reading(self, bg, t):
        self.history[t] = bg