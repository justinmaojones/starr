import numpy as np

from prefix_sum_tree._cython import get_prefix_sum_idx
from prefix_sum_tree._cython import update_prefix_sum_tree

class PrefixSumTree(np.ndarray):
        
    def __new__(self,shape_or_array,dtype=None):

        if isinstance(shape_or_array,PrefixSumTree):
            if dtype is None:
                return shape_or_array
            else:
                return shape_or_array.astype(dtype)
        
        elif isinstance(shape_or_array,np.ndarray):
            dtype = shape_or_array.dtype if dtype is None else dtype
            array = np.zeros(shape_or_array.shape,dtype=dtype).view(PrefixSumTree)
            array.ravel()[:] = shape_or_array.ravel()
            return array
        else:
            dtype = float if dtype is None else dtype
            return np.zeros(shape_or_array,dtype=dtype).view(PrefixSumTree)
            
    def __array_finalize__(self,array):
        
        if isinstance(self.base,PrefixSumTree):
            # inherit the same base and sum tree
            self._flat_base = self.base._flat_base
            self._indices = self.base._indices.reshape(array.shape)
            self._sumtree = self.base._sumtree
        else:
            # initialize
            self._flat_base = array.view(np.ndarray).ravel()
            self._indices = np.arange(array.size, dtype=np.int32).reshape(array.shape)
            self._sumtree = np.zeros_like(self._flat_base)
        
    def __array_wrap__(self, out_arr, context=None):
        # any op that transforms the array, other than setting values, 
        # should return an ndarray
        return super(PrefixSumTree, self).__array_wrap__(out_arr, context).view(np.ndarray)
    
    def __setitem__(self,idx,val):
        indices = np.ascontiguousarray(self._indices[idx]).ravel()
        values = np.ascontiguousarray(val,dtype=self._flat_base.dtype).ravel()
        update_prefix_sum_tree(
                indices, values, self._flat_base, self._sumtree)

    def __getitem__(self,idx):
        output = super(PrefixSumTree,self).__getitem__(idx)
        if self is output.base or self is output:
            # if the output is a view of self, then copy it
            # because working directly with a view of self is dangerous
            return output.view(np.ndarray).copy()
        else:
            # otherwise, already copied, so return a normal ndarray
            return output.view(np.ndarray)
    
    def get_prefix_sum_id(self,prefix_sum):
        # ensure prefix sum is the correct type and is contiguous
        prefix_sum = np.ascontiguousarray(prefix_sum,dtype=self.dtype)
        prefix_sum_flat = prefix_sum.ravel()
        # init return array
        output = np.zeros(prefix_sum.size,dtype=np.int32)
        # get ids
        get_prefix_sum_idx(output,prefix_sum_flat,self._flat_base,self._sumtree)
        return output.reshape(prefix_sum.shape)

    def sample(self,nsamples=1):
        # sample priority values in the cumulative sum
        vals = (self.sum() * np.random.rand(nsamples)).astype(self.dtype)
        # init return array
        output = np.zeros(nsamples,dtype=np.int32)
        # get sampled ids
        get_prefix_sum_idx(output,vals,self._flat_base,self._sumtree)
        return output

    def __sum__(self):
        if len(self) == 1:
            return self[0]
        else:
            return self._sumtree[1]

    def sum(self,*args,**kwargs):
        if len(args) == 0 and len(kwargs) == 0 and len(self) > 1:
            return self._sumtree[1]
        else:
            # TODO: sum with sumtree
            return super(PrefixSumTree, self).sum(*args,**kwargs)

