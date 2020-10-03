import numpy as np

from prefix_sum_tree._cython import get_prefix_sum_idx
from prefix_sum_tree._cython import strided_sum 
from prefix_sum_tree._cython import update_prefix_sum_tree

class PrefixSumTree(np.ndarray):
        
    def __new__(self,shape_or_array,dtype=None):

        if isinstance(shape_or_array,PrefixSumTree):
            if dtype is None:
                return shape_or_array
            else:
                return shape_or_array.astype(dtype)
        
        elif isinstance(shape_or_array,np.ndarray):
            if shape_or_array.size <= 1:
                raise ValueError("input to PrefixSumTree must have shape with at least 2 elements")
            assert shape_or_array.size > 1
            dtype = shape_or_array.dtype if dtype is None else dtype
            array = np.zeros(shape_or_array.shape,dtype=dtype).view(PrefixSumTree)
            array.ravel()[:] = shape_or_array.ravel()
            return array
        else:
            dtype = float if dtype is None else dtype
            array = np.zeros(shape_or_array,dtype=dtype)
            if array.size <= 1:
                raise ValueError("input to PrefixSumTree must have shape with at least 2 elements")
            return array.view(PrefixSumTree)
            
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

    # when a transformation is applied to a PrefixSumTree object, it is assumed that
    # we do not want a new PrefixSumTree object (which could result in a large
    # number of unwanted prefix sum tree updates)...and thus the transformation
    # is applied to the underlying array object, and an NDArray is returned
    def __array_prepare__(self, out_arr, context=None):
        if np.shares_memory(out_arr,self):
            return out_arr.view(np.ndarray).copy()
        else:
            return out_arr.view(np.ndarray)

    def __array_wrap__(self, out_arr, context=None):
        if np.shares_memory(out_arr,self):
            return out_arr.view(np.ndarray).copy()
        else:
            return out_arr.view(np.ndarray)

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
        flat_idx = np.zeros(nsamples,dtype=np.int32)
        # get sampled ids
        get_prefix_sum_idx(flat_idx,vals,self._flat_base,self._sumtree)
        # convert to array shape idx
        if self.ndim <= 1:
            return flat_idx
        else:
            return np.unravel_index(flat_idx,self.shape)

    def _parse_axis_arg(self,axis):
        if axis is None:
            return None
        else:
            axes = np.array(axis).reshape(-1)
            if len(set(axes)) < len(axes):
                raise ValueError("invalid axis argument: %s contains duplicates" % str(axis))
            if axes.max() > self.ndim:
                raise ValueError("invalid axis argument: %s is out of bounds" % str(axis))
            return np.arange(self.ndim)[axes]

    def sum(self,axis=None,keepdims=False):
        axes = self._parse_axis_arg(axis)
        if axes is None:
            if len(self) > 1:
                return self._sumtree[1]
            else:
                return super(PrefixSumTree, self).sum(keepdims=keepdims)
        else:
            if axes.min() == self.ndim - len(axes):
                # strides are contiguous along leaves of sumtree
                stride = int(np.prod(np.array(self.shape)[axes]))
                output = strided_sum(self._flat_base,self._sumtree,stride)
                if keepdims:
                    return output.reshape(list(output.shape)+[1]*len(axes))
                else:
                    return output
            else:
                return super(PrefixSumTree, self).sum(axis=axis,keepdims=keepdims)

