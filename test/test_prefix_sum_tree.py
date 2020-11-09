import unittest
import numpy as np
from prefix_sum_tree import PrefixSumTree
from prefix_sum_tree.prefix_sum_tree import _repr_with_new_class_name

class TestPrefixSumTree(unittest.TestCase):

    def test_array_creation_from_shape_inputs(self):
        prefix_sum_tree = PrefixSumTree(2)
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree.size,2)
        self.assertEqual(prefix_sum_tree._array.min(),0)
        self.assertEqual(prefix_sum_tree._array.max(),0)
        self.assertEqual(prefix_sum_tree._sumtree.min(),0)
        self.assertEqual(prefix_sum_tree._sumtree.max(),0)
        self.assertEqual(prefix_sum_tree.dtype,float)
        self.assertTrue(np.shares_memory(prefix_sum_tree._array, prefix_sum_tree._flat_base))

    def test_array_creation_from_shape_inputs_and_dtype(self):
        prefix_sum_tree = PrefixSumTree(2,dtype=int)
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree.size,2)
        self.assertEqual(prefix_sum_tree.dtype,int)
        self.assertEqual(prefix_sum_tree._array.min(),0)
        self.assertEqual(prefix_sum_tree._array.max(),0)
        self.assertEqual(prefix_sum_tree._sumtree.min(),0)
        self.assertEqual(prefix_sum_tree._sumtree.max(),0)
        self.assertEqual(prefix_sum_tree._array.dtype,int)
        self.assertEqual(prefix_sum_tree._flat_base.dtype,int)
        self.assertEqual(prefix_sum_tree._sumtree.dtype,int)
        self.assertTrue(np.shares_memory(prefix_sum_tree._array, prefix_sum_tree._flat_base))

    def test_array_creation_from_nd_array_input(self):
        input_array = np.array([0,1]).astype("int32")
        prefix_sum_tree = PrefixSumTree(input_array)
        # PrefixSumTree creates a new base array, and thus changes 
        # to input_array should not be reflected in prefix_sum_tree
        input_array[0] = 99
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree.size,2)
        self.assertEqual(prefix_sum_tree._array[0],0)
        self.assertEqual(prefix_sum_tree._array[1],1)
        self.assertEqual(prefix_sum_tree._sumtree[0],0)
        self.assertEqual(prefix_sum_tree._sumtree[1],1)
        self.assertEqual(prefix_sum_tree.dtype,np.dtype("int32"))
        self.assertEqual(prefix_sum_tree._array.dtype,np.dtype("int32"))
        self.assertEqual(prefix_sum_tree._flat_base.dtype,np.dtype("int32"))
        self.assertEqual(prefix_sum_tree._sumtree.dtype,np.dtype("int32"))
        self.assertFalse(np.shares_memory(prefix_sum_tree._array, input_array))
        self.assertFalse(np.shares_memory(prefix_sum_tree._flat_base, input_array))
        self.assertTrue(np.shares_memory(prefix_sum_tree._array, prefix_sum_tree._flat_base))

    def test_array_creation_from_nd_array_input_and_dtype(self):
        input_array = np.array([0,1]).astype("int32")
        prefix_sum_tree = PrefixSumTree(input_array,dtype="int64")
        # PrefixSumTree creates a new base array, and thus changes 
        # to input_array should not be reflected in prefix_sum_tree
        input_array[0] = 99
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree.size,2)
        self.assertEqual(prefix_sum_tree._array[0],0)
        self.assertEqual(prefix_sum_tree._array[1],1)
        self.assertEqual(prefix_sum_tree._sumtree[0],0)
        self.assertEqual(prefix_sum_tree._sumtree[1],1)
        self.assertEqual(prefix_sum_tree.dtype,np.dtype("int64"))
        self.assertFalse(np.shares_memory(prefix_sum_tree._array, input_array))
        self.assertFalse(np.shares_memory(prefix_sum_tree._flat_base, input_array))
        self.assertTrue(np.shares_memory(prefix_sum_tree._array, prefix_sum_tree._flat_base))

    def test_array_creation_from_prefix_sum_tree_input(self):
        input_prefix_sum_tree = PrefixSumTree(np.array([0,1]).astype("int32"))
        prefix_sum_tree = PrefixSumTree(input_prefix_sum_tree)
        # PrefixSumTree points to the same buffers as the input, and thus changes 
        # to input_prefix_sum_tree should be reflected in prefix_sum_tree
        self.assertEqual(id(input_prefix_sum_tree._array), id(prefix_sum_tree._array))
        self.assertEqual(id(input_prefix_sum_tree._flat_base), id(prefix_sum_tree._flat_base))
        self.assertEqual(id(input_prefix_sum_tree._indices), id(prefix_sum_tree._indices))
        self.assertEqual(id(input_prefix_sum_tree._sumtree), id(prefix_sum_tree._sumtree))
        input_prefix_sum_tree[0] = 99
        self.assertEqual(prefix_sum_tree[0],99)
        self.assertEqual(prefix_sum_tree[1],1)
        self.assertEqual(prefix_sum_tree._sumtree[0],0)
        self.assertEqual(prefix_sum_tree._sumtree[1],100)
        self.assertEqual(prefix_sum_tree.dtype,np.dtype("int32"))

    def test_array_creation_from_prefix_sum_tree_input_and_dtype(self):
        input_prefix_sum_tree = PrefixSumTree(np.array([0,1]).astype("int32"))
        prefix_sum_tree = PrefixSumTree(input_prefix_sum_tree,dtype="int64")
        # PrefixSumTree creates a new underlying prefix sum tree, because the type signature is different,
        # and thus changes to input_prefix_sum_tree should not be reflected in prefix_sum_tree
        input_prefix_sum_tree[0] = 99
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree.size,2)
        self.assertEqual(prefix_sum_tree._array[0],0)
        self.assertEqual(prefix_sum_tree._array[1],1)
        self.assertEqual(prefix_sum_tree._sumtree[0],0)
        self.assertEqual(prefix_sum_tree._sumtree[1],1)
        self.assertEqual(prefix_sum_tree.dtype,np.dtype("int64"))
        self.assertFalse(np.shares_memory(prefix_sum_tree._array, input_prefix_sum_tree))
        self.assertFalse(np.shares_memory(prefix_sum_tree._flat_base, input_prefix_sum_tree))
        self.assertTrue(np.shares_memory(prefix_sum_tree._array, prefix_sum_tree._flat_base))

    def test_invalid_array_creation(self):
        with self.assertRaises(ValueError):
            PrefixSumTree(0)
        with self.assertRaises(ValueError):
            PrefixSumTree(1)
        with self.assertRaises(ValueError):
            PrefixSumTree(np.array(1))
        with self.assertRaises(ValueError):
            PrefixSumTree(np.array([1]))

    def test_reshape(self):
        x1 = PrefixSumTree(np.array([0,1,2,3]).astype("int32"))
        x2 = x1.reshape((2,2))
        # when reshaping, the underlying data, index, and sumtree objects are shared
        self.assertNotEqual(id(x1), id(x2))
        self.assertEqual(id(x1._flat_base), id(x2._flat_base))
        self.assertEqual(id(x1._sumtree), id(x2._sumtree))
        self.assertEqual(id(x1._indices.base), id(x2._indices.base))
        # and thus changes in x1 will be reflected in x2, and vice versa
        x1[0] = 10
        self.assertEqual(x1[0],10)
        self.assertEqual(x1._sumtree[1],16)
        self.assertEqual(x2[0,0],10)
        self.assertEqual(x2._sumtree[1],16)
        x2[1] = 20
        self.assertEqual(x1[2],20)
        self.assertEqual(x1[3],20)
        self.assertEqual(x1._sumtree[1],51)
        self.assertEqual(x2[1,0],20)
        self.assertEqual(x2[1,1],20)
        self.assertEqual(x2._sumtree[1],51)

    def test_ravel(self):
        x = PrefixSumTree(np.array([[0,1],[2,3]]).astype("int32"))
        x2 = x.ravel()
        self.assertTrue(np.shares_memory(x,x2))
        self.assertEqual(x.ndim,2)
        self.assertEqual(x2.ndim,1)
        self.assertEqual(x2.size,4)

    def test_ufunc(self):
        x = PrefixSumTree(np.array([4,9]),dtype='float64')
        y = np.sqrt(x)
        self.assertEqual(y[0],2)
        self.assertEqual(y[1],3)
        y = np.sqrt(x,dtype='float32')
        self.assertEqual(y[0],2)
        self.assertEqual(y[1],3)

    def test_other_basic_math_funcs(self):
        x = PrefixSumTree(np.array([0,1,2,3]).astype("int32"))
        self.assertEqual(x.min(),0)
        self.assertEqual(x.max(),3)

    def test_ndim(self):
        x = PrefixSumTree(np.array([0,1,2,3]).astype("int32"))
        self.assertEqual(x.ndim,1)
        x = PrefixSumTree(np.array([[0,1],[2,3]]).astype("int32"))
        self.assertEqual(x.ndim,2)

    def test_size(self):
        x = PrefixSumTree(np.array([0,1,2,3]).astype("int32"))
        self.assertEqual(x.size,4)
        x = PrefixSumTree(np.array([[0,1],[2,3]]).astype("int32"))
        self.assertEqual(x.size,4)

    def test_set(self):
        # [[0,1],
        #  [2,3],
        #  [4,5]]
        x = PrefixSumTree(np.array([0,1,2,3,4,5]).astype("int32").reshape((3,2)))
        # normal array indexing
        x[1] = 10
        self.assertEqual(x[0,0],0)
        self.assertEqual(x[0,1],1)
        self.assertEqual(x[1,0],10)
        self.assertEqual(x[1,1],10)
        self.assertEqual(x[2,0],4)
        self.assertEqual(x[2,1],5)
        self.assertEqual(x._sumtree[1],30)
        # array of indexes 
        x[np.array([0,1]),np.array([1,0])] = 20
        self.assertEqual(x[0,0],0)
        self.assertEqual(x[0,1],20)
        self.assertEqual(x[1,0],20)
        self.assertEqual(x[1,1],10)
        self.assertEqual(x[2,0],4)
        self.assertEqual(x[2,1],5)
        self.assertEqual(x._sumtree[1],59)
        # boolean indexing
        x[np.array([True,False,True])] = 30
        self.assertEqual(x[0,0],30)
        self.assertEqual(x[0,1],30)
        self.assertEqual(x[1,0],20)
        self.assertEqual(x[1,1],10)
        self.assertEqual(x[2,0],30)
        self.assertEqual(x[2,1],30)
        self.assertEqual(x._sumtree[1],150)

    def test_get(self):
        # [[0,1],
        #  [2,3],
        #  [4,5]]
        x = PrefixSumTree(np.array([0,1,2,3,4,5]).astype("int32").reshape((3,2)))
        # normal array indexing
        y = x[1]
        self.assertFalse(isinstance(y,PrefixSumTree))
        self.assertFalse(np.shares_memory(x,y))
        self.assertEqual(y[0],2)
        self.assertEqual(y[1],3)
        self.assertEqual(y.size,2)
        # normal array indexing
        y = x[1:3,1:]
        self.assertFalse(isinstance(y,PrefixSumTree))
        self.assertFalse(np.shares_memory(x,y))
        self.assertEqual(y[0],3)
        self.assertEqual(y[1],5)
        self.assertEqual(y.size,2)
        # array of indexes 
        y = x[np.array([0,1]),np.array([1,0])]
        self.assertFalse(isinstance(y,PrefixSumTree))
        self.assertFalse(np.shares_memory(x,y))
        self.assertEqual(y[0],1)
        self.assertEqual(y[1],2)
        self.assertEqual(y.size,2)
        # boolean indexing
        y = x[np.array([True,False,True])]
        self.assertFalse(isinstance(y,PrefixSumTree))
        self.assertFalse(np.shares_memory(x,y))
        self.assertEqual(y[0,0],0)
        self.assertEqual(y[0,1],1)
        self.assertEqual(y[1,0],4)
        self.assertEqual(y[1,1],5)
        self.assertEqual(y.size,4)

    def test_get_prefix_sum_id(self):
        # This validates pipes are setup correctly.
        # Does not test full functionality of get_prefix_sum_idx, since that is
        # tested elsewhere.
        x = PrefixSumTree(np.array([1,2,3,4]).astype("int32"))
        # single element
        y =  x.get_prefix_sum_id(1.5)
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertEqual(y[0],1)
        self.assertEqual(y.size,1)
        # array of sums
        y =  x.get_prefix_sum_id([[0.5,1.5],[3.5,6.5]],flatten_indices=False)
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertEqual(y[0,0],0)
        self.assertEqual(y[0,1],1)
        self.assertEqual(y[1,0],2)
        self.assertEqual(y[1,1],3)
        self.assertEqual(y.size,4)
        # array of sums (doesn't change when flatten_indices=True)
        y =  x.get_prefix_sum_id([[0.5,1.5],[3.5,6.5]],flatten_indices=True)
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertEqual(y[0,0],0)
        self.assertEqual(y[0,1],1)
        self.assertEqual(y[1,0],2)
        self.assertEqual(y[1,1],3)
        self.assertEqual(y.size,4)
        # works for 2d PrefixSumTree
        x = PrefixSumTree(np.array([[1,2],[3,4]]).astype("int32"))
        # single element
        y =  x.get_prefix_sum_id(1.5,flatten_indices=False)
        self.assertTrue(isinstance(y,tuple))
        self.assertEqual(len(y),2)
        self.assertTrue(isinstance(y[0],np.ndarray))
        self.assertTrue(isinstance(y[1],np.ndarray))
        self.assertFalse(isinstance(y[0],PrefixSumTree))
        self.assertFalse(isinstance(y[1],PrefixSumTree))
        self.assertEqual(y[0][0],0)
        self.assertEqual(y[0].size,1)
        self.assertEqual(y[1][0],1)
        self.assertEqual(y[1].size,1)
        # single element
        y =  x.get_prefix_sum_id(1.5,flatten_indices=True)
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertEqual(y[0],1)
        self.assertEqual(y.size,1)
        # flat array of sums
        y =  x.get_prefix_sum_id([0.5,1.5,3.5,6.5],flatten_indices=True)
        self.assertEqual(y[0],0)
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertEqual(y[1],1)
        self.assertEqual(y[2],2)
        self.assertEqual(y[3],3)
        self.assertEqual(y.size,4)
        # 2d array of sums
        y =  x.get_prefix_sum_id([[0.5,1.5],[3.5,6.5]],flatten_indices=True)
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertEqual(y[0,0],0)
        self.assertEqual(y[0,1],1)
        self.assertEqual(y[1,0],2)
        self.assertEqual(y[1,1],3)
        self.assertEqual(y.size,4)
        # returns tuple of flat array of indices 
        y =  x.get_prefix_sum_id([0.5,1.5,3.5,6.5],flatten_indices=False)
        self.assertTrue(isinstance(y,tuple))
        self.assertEqual(len(y),2)
        self.assertTrue(isinstance(y[0],np.ndarray))
        self.assertTrue(isinstance(y[1],np.ndarray))
        self.assertFalse(isinstance(y[0],PrefixSumTree))
        self.assertFalse(isinstance(y[1],PrefixSumTree))
        self.assertEqual(y[0][0],0)
        self.assertEqual(y[0][1],0)
        self.assertEqual(y[0][2],1)
        self.assertEqual(y[0][3],1)
        self.assertEqual(y[0].size,4)
        self.assertEqual(y[1][0],0)
        self.assertEqual(y[1][1],1)
        self.assertEqual(y[1][2],0)
        self.assertEqual(y[1][3],1)
        self.assertEqual(y[1].size,4)
        # returns tuple of 2d array of indices 
        y =  x.get_prefix_sum_id([[0.5,1.5],[3.5,6.5]],flatten_indices=False)
        self.assertTrue(isinstance(y,tuple))
        self.assertEqual(len(y),2)
        self.assertTrue(isinstance(y[0],np.ndarray))
        self.assertTrue(isinstance(y[1],np.ndarray))
        self.assertFalse(isinstance(y[0],PrefixSumTree))
        self.assertFalse(isinstance(y[1],PrefixSumTree))
        self.assertEqual(y[0][0,0],0)
        self.assertEqual(y[0][0,1],0)
        self.assertEqual(y[0][1,0],1)
        self.assertEqual(y[0][1,1],1)
        self.assertEqual(y[0].size,4)
        self.assertEqual(y[1][0,0],0)
        self.assertEqual(y[1][0,1],1)
        self.assertEqual(y[1][1,0],0)
        self.assertEqual(y[1][1,1],1)
        self.assertEqual(y[1].size,4)

    def test_sample(self):
        # flat index
        x = PrefixSumTree(np.array([1,2,3,4]).astype("int32"))
        # single sample
        idx = x.sample(flatten_indices=False)
        self.assertTrue(isinstance(idx,np.ndarray))
        self.assertEqual(idx.size,1)
        self.assertEqual(x[idx].size,1)
        # single sample
        idx = x.sample(flatten_indices=True)
        self.assertTrue(isinstance(idx,np.ndarray))
        self.assertEqual(idx.size,1)
        self.assertEqual(x[idx].size,1)
        # multiple samples
        idx = x.sample(20,flatten_indices=False)
        self.assertTrue(isinstance(idx,np.ndarray))
        self.assertEqual(x[idx].size,20)
        # multiple samples and flatten_indices=True doesn't change result
        idx = x.sample(20,flatten_indices=True)
        self.assertTrue(isinstance(idx,np.ndarray))
        self.assertEqual(x[idx].size,20)
        # 2d index
        x = PrefixSumTree(np.array([[1,2],[3,4]]).astype("int32"))
        # single sample
        idx = x.sample(flatten_indices=False)
        self.assertTrue(isinstance(idx,tuple))
        self.assertEqual(len(idx),2)
        self.assertTrue(isinstance(idx[0],np.ndarray))
        self.assertTrue(isinstance(idx[1],np.ndarray))
        self.assertEqual(x[idx].size,1)
        # multiple samples
        idx = x.sample(20,flatten_indices=False)
        self.assertTrue(isinstance(idx,tuple))
        self.assertEqual(len(idx),2)
        self.assertTrue(isinstance(idx[0],np.ndarray))
        self.assertTrue(isinstance(idx[1],np.ndarray))
        self.assertEqual(x[idx].size,20)
        # multiple samples and flatten_indices=True returns flattened indices
        idx = x.sample(20,flatten_indices=True)
        self.assertTrue(isinstance(idx,np.ndarray))
        self.assertEqual(len(idx),20)
        self.assertEqual(idx.ndim,1)

    def test_sum(self):
        # 1d array
        x = PrefixSumTree(np.array([1,2,3,4]).astype("int32"))
        self.assertEqual(x.sum(),10)
        # 2d array
        x = PrefixSumTree(np.array([[1,2],[3,4]]).astype("int32"))
        self.assertEqual(x.sum(),10)
        self.assertEqual(x.sum(keepdims=True).ndim,2)
        self.assertEqual(x.sum(keepdims=True)[0,0],10)
        y = x.sum(axis=1)
        self.assertEqual(y[0],3)
        self.assertEqual(y[1],7)
        self.assertEqual(y.size,2)
        self.assertEqual(y.ndim,1)
        y = x.sum(axis=1,keepdims=True)
        self.assertEqual(y[0,0],3)
        self.assertEqual(y[1,0],7)
        self.assertEqual(y.size,2)
        self.assertEqual(y.ndim,2)


    def test_sum_with_args(self):
        v = np.random.choice(100,size=(100,100))
        x = PrefixSumTree(v)
        self.assertFalse(isinstance(x.sum(axis=1),PrefixSumTree))
        self.assertTrue(isinstance(x.sum(axis=1),np.ndarray))
        self.assertFalse(isinstance(x.sum(axis=1,keepdims=True),PrefixSumTree))
        self.assertTrue(isinstance(x.sum(axis=1,keepdims=True),np.ndarray))
        self.assertEqual(0, np.abs(x.sum(axis=1) - v.sum(axis=1)).max())
        self.assertEqual(0, np.abs(x.sum(axis=0) - v.sum(axis=0)).max())
        self.assertEqual(0, np.abs(x.sum(axis=1,keepdims=True) - v.sum(axis=1,keepdims=True)).max())
        self.assertEqual(0, np.abs(x.sum(axis=0,keepdims=True) - v.sum(axis=0,keepdims=True)).max())
        self.assertEqual(0,np.abs(x.sum() - v.sum()).max())

    def test_parse_axis_arg(self):
        x = PrefixSumTree((2,3,4))

        def check_normal_inputs(axis, expected):
            output = x._parse_axis_arg(axis)
            self.assertTrue(isinstance(output, np.ndarray))
            self.assertEqual(output.ndim, 1)
            self.assertEqual(len(output), len(expected))
            self.assertTrue(np.issubdtype(output.dtype, np.integer))
            self.assertTrue(np.all(output==expected))

        self.assertEqual(x._parse_axis_arg(None), None)
        check_normal_inputs(0, np.array([0], dtype=int))
        check_normal_inputs((0,1), np.array([0,1], dtype=int))
        check_normal_inputs((1,-1), np.array([1,2], dtype=int))
        check_normal_inputs(-1, np.array([2], dtype=int))
        check_normal_inputs(-2, np.array([1], dtype=int))

        with self.assertRaises(IndexError):
            x._parse_axis_arg((1,1))
        with self.assertRaises(IndexError):
            x._parse_axis_arg(3)

    def test_invalid_entry_error(self):
        with self.assertRaises(ValueError):
            PrefixSumTree(np.array([-1,0,1]).astype("int32"))
        with self.assertRaises(ValueError):
            x = PrefixSumTree(np.array([1,2,3,4]).astype("int32"))
            x[:2] = -1

    def test_sample_validation(self):
        with self.assertRaises(ValueError):
            x = PrefixSumTree(10)
            x.sample(10)

    def test_array(self):
        x = PrefixSumTree(np.arange(4))
        # copy
        y = x.array()
        self.assertFalse(isinstance(y,PrefixSumTree))
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertFalse(np.shares_memory(y,x._array))
        # shared memory
        y = x.array(copy=False)
        self.assertFalse(isinstance(y,PrefixSumTree))
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertTrue(np.shares_memory(y,x._array))

    def test_sumtree(self):
        x = PrefixSumTree(np.arange(4))
        # copy
        y = x.sumtree()
        self.assertFalse(isinstance(y,PrefixSumTree))
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertFalse(np.shares_memory(y,x._sumtree))
        # shares memory
        y = x.sumtree(copy=False)
        self.assertFalse(isinstance(y,PrefixSumTree))
        self.assertTrue(isinstance(y,np.ndarray))
        self.assertTrue(np.shares_memory(y,x._sumtree))
        
    def test_repr(self):
        x = PrefixSumTree(np.array([1,2,3,4]).reshape(2,2),dtype='float32')
        self.assertEqual(repr(x), 'PrefixSumTree([[1., 2.],\n               [3., 4.]], dtype=float32)')
        self.assertEqual(_repr_with_new_class_name('A',x._array), 'A([[1., 2.],\n   [3., 4.]], dtype=float32)')
        self.assertEqual(_repr_with_new_class_name('array',x._array),repr(x._array))

        
if __name__ == '__main__':
    unittest.main()
