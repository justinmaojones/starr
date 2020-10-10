import unittest
import numpy as np
from prefix_sum_tree import PrefixSumTree

class TestPrefixSumTree(unittest.TestCase):

    def test_array_creation_from_shape_inputs(self):
        prefix_sum_tree = PrefixSumTree(2)
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree.size,2)
        self.assertEqual(prefix_sum_tree.min(),0)
        self.assertEqual(prefix_sum_tree.max(),0)
        self.assertEqual(prefix_sum_tree.dtype,float)

    def test_array_creation_from_shape_inputs_and_dtype(self):
        prefix_sum_tree = PrefixSumTree(2,dtype=int)
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree.dtype,int)

    def test_array_creation_from_nd_array_input(self):
        input_array = np.array([0,1]).astype("int32")
        prefix_sum_tree = PrefixSumTree(input_array)
        # PrefixSumTree creates a new base array, and thus changes 
        # to input_array should not be reflected in prefix_sum_tree
        input_array[0] = 99
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree[0],0)
        self.assertEqual(prefix_sum_tree[1],1)
        self.assertEqual(prefix_sum_tree.dtype,np.dtype("int32"))

    def test_array_creation_from_nd_array_input_and_dtype(self):
        input_array = np.array([0,1]).astype("int32")
        prefix_sum_tree = PrefixSumTree(input_array,dtype="int64")
        # PrefixSumTree creates a new base array, and thus changes 
        # to input_array should not be reflected in prefix_sum_tree
        input_array[0] = 99
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree[0],0)
        self.assertEqual(prefix_sum_tree[1],1)
        self.assertEqual(prefix_sum_tree.dtype,np.dtype("int64"))

    def test_array_creation_from_prefix_sum_tree_input(self):
        input_prefix_sum_tree = PrefixSumTree(np.array([0,1]).astype("int32"))
        prefix_sum_tree = PrefixSumTree(input_prefix_sum_tree)
        # PrefixSumTree returns the input, and thus changes 
        # to input_prefix_sum_tree should be reflected in prefix_sum_tree
        self.assertEqual(id(input_prefix_sum_tree), id(prefix_sum_tree))
        input_prefix_sum_tree[0] = 99
        self.assertEqual(prefix_sum_tree[0],99)
        self.assertEqual(prefix_sum_tree[1],1)
        self.assertEqual(prefix_sum_tree._sumtree[1],100)
        self.assertEqual(prefix_sum_tree.dtype,np.dtype("int32"))

    def test_array_creation_from_prefix_sum_tree_input_and_dtype(self):
        input_prefix_sum_tree = PrefixSumTree(np.array([0,1]).astype("int32"))
        prefix_sum_tree = PrefixSumTree(input_prefix_sum_tree,dtype="int64")
        # PrefixSumTree creates a new underlying prefix sum tree, because the type signature is different,
        # and thus changes to input_prefix_sum_tree should not be reflected in prefix_sum_tree
        input_prefix_sum_tree[0] = 99
        self.assertTrue(isinstance(prefix_sum_tree,PrefixSumTree))
        self.assertEqual(prefix_sum_tree[0],0)
        self.assertEqual(prefix_sum_tree[1],1)
        self.assertEqual(prefix_sum_tree.dtype,np.dtype("int64"))

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
        self.assertEqual(x1[1],20)
        self.assertEqual(x1._sumtree[1],35)
        self.assertEqual(x2[0,1],20)
        self.assertEqual(x2._sumtree[1],35)

    def test_transformation_does_not_affect_prefix_sum_tree(self):
        x = PrefixSumTree(np.array([0,1,2,3]).astype("int32"))
        x2 = x+1
        # when a transformation is applied to a PrefixSumTree object, it is assumed that
        # we do not want a new PrefixSumTree object (which could result in a large
        # number of unwanted prefix sum tree updates)...and thus the transformation
        # is applied to the underlying array object, and an NDArray is returned
        self.assertFalse(isinstance(x2,PrefixSumTree))

        # underlying x and sumtree is unchanged
        self.assertEqual(x2[0],1)
        self.assertEqual(x[0],0)
        self.assertEqual(x._sumtree[1],6)

        # ditto for in-place assignment operators
        x3 = x
        x3 += 10
        self.assertTrue(isinstance(x,PrefixSumTree))
        self.assertFalse(isinstance(x3,PrefixSumTree))
        self.assertEqual(x3[0],10)
        self.assertEqual(x[0],0)
        self.assertEqual(x._sumtree[1],6)

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
        self.assertEqual(y[0],1)
        self.assertEqual(y.size,1)
        # array of sums
        y =  x.get_prefix_sum_id([[0.5,1.5],[3.5,6.5]])
        self.assertEqual(y[0,0],0)
        self.assertEqual(y[0,1],1)
        self.assertEqual(y[1,0],2)
        self.assertEqual(y[1,1],3)
        self.assertEqual(y.size,4)

    def test_sample(self):
        # flat index
        x = PrefixSumTree(np.array([1,2,3,4]).astype("int32"))
        # single sample
        idx = x.sample()
        self.assertTrue(isinstance(idx,np.ndarray))
        self.assertEqual(idx.size,1)
        self.assertEqual(x[idx].size,1)
        # multiple samples
        idx = x.sample(20)
        self.assertTrue(isinstance(idx,np.ndarray))
        self.assertEqual(x[idx].size,20)
        # 2d index
        x = PrefixSumTree(np.array([[1,2],[3,4]]).astype("int32"))
        # single sample
        idx = x.sample()
        self.assertTrue(isinstance(idx,tuple))
        self.assertEqual(len(idx),2)
        self.assertTrue(isinstance(idx[0],np.ndarray))
        self.assertTrue(isinstance(idx[1],np.ndarray))
        self.assertEqual(x[idx].size,1)
        # multiple samples
        idx = x.sample(20)
        self.assertTrue(isinstance(idx,tuple))
        self.assertEqual(len(idx),2)
        self.assertTrue(isinstance(idx[0],np.ndarray))
        self.assertTrue(isinstance(idx[1],np.ndarray))
        self.assertEqual(x[idx].size,20)

    def test_sum(self):
        x = PrefixSumTree(np.array([1,2,3,4]).astype("int32"))
        self.assertEqual(x.sum(),10)
        x = PrefixSumTree(np.array([[1,2],[3,4]]).astype("int32"))
        self.assertEqual(x.sum(),10)
        y = x.sum(axis=1)
        self.assertEqual(y[0],3)
        self.assertEqual(y[1],7)
        self.assertEqual(y.size,2)

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
        
if __name__ == '__main__':
    unittest.main()
