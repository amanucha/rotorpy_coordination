import numpy as np

class param:
    def __init__(self,L_size):
        if L_size == 2:
            self.L = np.matrix([[1, -1], [-1, 1]])
        elif L_size == 3:
            self.L = np.matrix([[2, -1, -1], [-1, 2, -1],[-1, -1, 2]])
        elif L_size == 4:
            self.L = np.matrix([[3, -1, -1, -1], [-1, 3, -1, -1],[-1, -1, 3, -1],[-1, -1, -1, 3]])
        else:
            self.L = np.matrix([[1, -1], [-1, 1]])  

        self.a = 1.5 # 1.0
        self.b = 3.6 # 128.0
        self.delta = 3.0
        self.step = 0.02 # [s] 50Hz # 0.05 # [s] 20Hz
        self.d_min = 2.0 #1.8 # [m]    
        self.d_min_2 = 1.8 #1.35 #[m]
        
