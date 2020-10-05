import numpy as np

class SumTree(object):
    
    def __init__(self,size):
        assert size > 0, "size must be greater than 0"
        self._size = size
        self._build_data_object()
        
    def _build_data_object(self):
        self._data = [0]*(2*self._size)

    def __setitem__(self, idx, val):
        assert idx >= 0 and idx < self._size, "idx must be in [0,%d)" % self._size
        assert val >= 0, "val must be >= 0"
        idx += self._size
        diff = val - self._data[idx]
        while idx > 0:
            self._data[idx] += diff
            idx //= 2
            
    def __getitem__(self,idx):
        return self._data[idx+self._size]

    def __len__(self):
        return self._size

    def sum(self):
        return self._data[1]
    
    def get_prefix_sum_idx(self,val):
        assert val <= self.sum()
        i = 1
        while i < self._size:
            left = 2*i
            right = 2*i+1
            if val >= self._data[left]:
                i = right
                val -= self._data[left]
            else:
                i = left
            
        return i - self._size
    
    def sample(self,n=1):
        x = np.random.rand(n)*self.sum()
        if n == 1:
            return self.get_prefix_sum_idx(x[0])
        else:
            return [self.get_prefix_sum_idx(v) for v in x]

class MinTree(object):
    
    def __init__(self,size):
        assert size > 0, "size must be greater than 0"
        self._size = size
        self._build_data_object()
        
    def _build_data_object(self):
        self._data = [np.inf]*(2*self._size)
        
    def __setitem__(self, idx, val):
        assert idx >= 0 and idx < self._size, "idx must be in [0,%d)" % self._size
        idx += self._size
        self._data[idx] = val
        idx //= 2
        while idx > 0:
            self._data[idx] = min(self._data[2*idx],self._data[2*idx+1])
            idx //= 2
            
    def __getitem__(self,idx):
        return self._data[idx+self._size]

    def __len__(self):
        return self._size

    def min(self):
        return self._data[1]

class CircularSumTree(SumTree):

    def __init__(self,*args,**kwargs):
        super(CircularSumTree,self).__init__(*args,**kwargs)
        self._index = 0
        
    def append(self,val):
        self[self._index] = val
        self._index = (self._index + 1) % self._size

class CircularMinTree(MinTree):

    def __init__(self,*args,**kwargs):
        super(CircularMinTree,self).__init__(*args,**kwargs)
        self._index = 0
        
    def append(self,val):
        self[self._index] = val
        self._index = (self._index + 1) % self._size
