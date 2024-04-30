
class randgen():
    def __init__(self,seed, a = 1103515245, c = 12345 ,m = 32768, interval = (0,1), integer = False, decimals = None):
        #initiation of data input, seed , other LCG parameters, and the interval in which you need the random numbers in.

        self.term = seed
        self.a = a
        self.c = c
        self.m = m
        self.decimals = decimals
        self.integer = integer
        if interval[0] > interval[1]:
            self.interval = (interval[1],interval[0])
        elif interval[0] == interval[1]:
            print('Invalid interval for LCG')
        else:
            self.interval = interval

    def gen(self):
        #generates a random number in the range (0,1)
        self.term = (((self.a * self.term) + self.c) % self.m)
        value = (((self.interval[1]-self.interval[0])*(self.term / self.m)) + self.interval[0])
        if self.integer is True:
            value = int(value)
        if self.decimals is not None:
            value = round(value,self.decimals)
        return value

    def genlist(self,length):
        # returns a list of 'n' random numbers in the range (0,1) where 'n' is 'length'.
        RNs = []
        for i in range(length):
            self.term = (((self.a * self.term) + self.c) % self.m)
            value = (((self.interval[1]-self.interval[0])*(self.term / self.m)) + self.interval[0])
            if self.integer is True:
                value = int(value)
            if self.decimals is not None:
                value = round(value,self.decimals)  
                
            RNs.append(value)
        return RNs
    
    def non_uniform_inverse(self, inverse_cdf_func, length = 1, expected_interval = (0,1), cdf_func = None):
        #returns the inverse of the non-uniform distribution function at x
        normalrange_flag = False
        if expected_interval and self.interval != (0,1):
            curr_interval = self.interval
            normalrange_flag = True
            self.interval = (0,1)
        
        
        ### code responsible for generating in a specific range from transform fn
        expectedrange_flag = False
        if cdf_func is not None:
            rn_interval = (cdf_func(expected_interval[0]), cdf_func(expected_interval[1]))
            curr_interval2 = self.interval
            self.interval = rn_interval
            expectedrange_flag = True
             
        curr_length = 0
        vals = []
        while curr_length < length:
            x = self.gen()
            val = inverse_cdf_func(x)
            
            ### another code responsible for generating in a specific range directly
            if val < expected_interval[0] or val > expected_interval[1]:
                continue
            
            
            vals.append(val)
            curr_length += 1
        
        if expectedrange_flag:  
            self.interval = curr_interval2
        if normalrange_flag:
            self.interval = curr_interval
            
        if length == 1:
            return val
        else:
            return vals
        