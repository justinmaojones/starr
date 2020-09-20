import unittest
import numpy as np
from prefix_sum_tree._cython import get_prefix_sum_idx
from prefix_sum_tree._cython import update_prefix_sum_tree

class TestCythonPrefixSumTree(unittest.TestCase):

    def test_sum_tree(self):

        #         [25]       
        #        /    \
        #     [16]    [9]
        #     /  \    / \
        #   [13]  3  4   5    
        #   /  \
        #  6    7

        indices = np.array([0,1,2,3,4],dtype=int)
        values = np.array([3,4,5,6,7],dtype=float)
        base = np.zeros_like(values)
        sum_tree = np.zeros_like(values)

        # initially populate with some random positive values
        update_prefix_sum_tree(
            indices, np.random.randint(1,10,5).astype(float), base, sum_tree)

        # update tree with values
        update_prefix_sum_tree(
            indices, values, base, sum_tree)

        # leaves should be equal to values
        for i in range(len(base)):
            self.assertEqual(base[i], values[i])

        # value at index 0 should always be zero
        self.assertEqual(sum_tree[0], 0)

        # root node of sum_tree should be sum of all values 
        self.assertEqual(sum_tree[1], values.sum())

        # each node in sum_tree should be sum of its children
        self.assertEqual(sum_tree[1], sum_tree[2] + sum_tree[3])
        self.assertEqual(sum_tree[2], sum_tree[4] + values[0])
        self.assertEqual(sum_tree[3], values[1] + values[2])
        self.assertEqual(sum_tree[4], values[3] + values[4])

        # base and values_to_search are made of integers to check for edge cases
        values_to_search = np.array([0,5,6,12,13,15,16,19,20,24,25],dtype=float)
        expected_result = np.array([3,3,4,4,0,0,1,1,2,2,2],dtype=int)
        output = np.zeros_like(values_to_search).astype(int)
        get_prefix_sum_idx(output,values_to_search,base,sum_tree)

        for v,e in zip(output,expected_result):
            self.assertEqual(v,e)


if __name__ == '__main__':
    unittest.main()
