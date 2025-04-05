from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from sortedcontainers import SortedDict

class CGM:
    # in the future, want to be able to configure 
    def __init__(self, config=None):
        self.t = 0
        self.history: Dict[int, float]= SortedDict()
        self.config = config
     
    def get_reading(self, raw):
        bias = self.config.bias
        offset = self.config.offset
        return int(raw * bias + offset)
        
    def set_config(self, config):
        self.config = config
        
class DexcomCGM(CGM):
    def get_reading(self, t):
        t = t - t % 5 # sampling time is 5 min
        return int(self.history[t])
    
    def post_reading(self, bg, t):
        if t % 5 == 0:
            self.history[t] = bg
       
       
# https://www.mdpi.com/1424-8220/19/23/5320
 
class VettorettiCGM(CGM):
    
    # error has two components: calibration error and 
    def __init__(self, config={}):
        super().__init__(config)
        
        # use median parameters from two-step fitting process
        
        self.a0 = 0.95
        self.a1 = 0.002
        self.a2 = 0
        self.b0 = 7.30
        self.alph1 = 1.3
        self.alph2 = -0.46
        self.sigma = 3.20
        self.errors = {-5: 0, -10: 0} # hack to make error(0) and error(5) work
        self.last_reading = None
        self.start_day = config.get('start_day', 0)
    
    def get_reading(self, t):
        if t % 5 != 0:
            return self.last_reading
        IG = self.history[t]
        days = t / 1440 + self.start_day # t is initially in minutes
        IG_s = (self.a0 + self.a1 * days + self.a2 * days**2) * IG + self.b0
        AR_error = self.alph1 * self.errors[t - 5] + self.alph2 * self.errors[t - 10] + np.random.normal(scale=self.sigma)
        self.errors[t] = AR_error
        CGM = IG_s + AR_error
        self.last_reading = int(CGM)
        return int(CGM)
        

        
class BasalAttackCGM(CGM):
    def get_reading(self, t):
        real = super().get_reading(t)
        if t <= 15:
            return real
        if real <= 90:
            return 81
        return real
    
    
if __name__ == '__main__':
    real = [120, 125, 126, 131, 128, 124, 124, 124, 124, 124, 124, 124, 124, 124]
    cgm = VettorettiCGM()
    min_reading = 120
    for i in range(1440 * 10 // 5):
        cgm.post_reading(120, i * 5)
        min_reading = min(min_reading, cgm.get_reading(i * 5))
        print(f'error(t = {i * 5}) = {cgm.get_reading(i * 5)}')
    print(min_reading)
    
        