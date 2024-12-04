class CGM:
    # in the future, want to be able to configure 
    def __init__(self, config):
        self.t = 0
        self.history = []
    
    def get_reading(self):
        pass
    
    def post_reading(self, bg):
        self.history.append((self.t, bg))
    
    def delay(self, seconds):
        self.t += seconds