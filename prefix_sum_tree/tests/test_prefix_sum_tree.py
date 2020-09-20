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
        # is applied to the underlying array object
        self.assertTrue(isinstance(x2,np.ndarray))
        # underlying x and sumtree is unchanged
        self.assertEqual(x2[0],1)
        self.assertEqual(x[0],0)
        self.assertEqual(x._sumtree[1],6)
        # ditto for +=
        x3 = x
        x3 += 10
        self.assertTrue(isinstance(x,PrefixSumTree))
        self.assertTrue(isinstance(x3,np.ndarray))
        self.assertEqual(x3[0],10)
        self.assertEqual(x[0],10)
        self.assertEqual(x._sumtree[1],6)


        
if __name__ == '__main__':
    unittest.main()
