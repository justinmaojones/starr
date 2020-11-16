import unittest
import numpy as np
from sumtree_array.experimental.slow import SumTree, MinTree


class TestSumTree(unittest.TestCase):
    def _round_up_to_nearest_power_of_two(self, x):
        y = x
        k = 0
        while y > 0:
            y >>= 1
            k += 1
        return 2 ** k

    def test_round_up_to_nearest_power_of_two(self):
        self.assertEqual(self._round_up_to_nearest_power_of_two(1), 2)
        self.assertEqual(self._round_up_to_nearest_power_of_two(2), 4)
        self.assertEqual(self._round_up_to_nearest_power_of_two(3), 4)
        self.assertEqual(self._round_up_to_nearest_power_of_two(4), 8)

    def run_test_get_prefix_sum_idx(self, x, sum_tree):
        # test get_prefix_sum_idx
        n = len(sum_tree)
        n_up = self._round_up_to_nearest_power_of_two(n)
        # n_up = (n//2)*4 # round up to nearest power of 2
        j = (n_up - n) % n  # offset due to array length not being power of 2
        xsum = x[j]
        for i in range(int(sum(x))):
            if i >= xsum:
                j = (j + 1) % n
                xsum += x[j]
            self.assertEqual(j, sum_tree.get_prefix_sum_idx(i))

    def test_sum_tree(self):

        for size in range(2, 16):
            x = np.random.choice(20, size=size)
            x += 1  # ensure positive
            sum_tree = SumTree(size)

            for i, v in enumerate(x):
                sum_tree[i] = v

            # test set
            for i in range(size):
                self.assertEqual(sum_tree[i], x[i])

            # test sum
            self.assertEqual(sum_tree.sum(), x.sum())

            # test get_prefix_sum_idx
            self.run_test_get_prefix_sum_idx(x, sum_tree)

    def test_sum_tree_after_multiple_sets(self):

        for size in range(2, 16):
            x = np.random.choice(50, size=size)
            x += 1  # ensure positive
            sum_tree = SumTree(size)

            for i, v in enumerate(x):
                sum_tree[i] = v

            x = np.random.choice(50, size=size)
            x += 1  # ensure positive
            for i, v in enumerate(x):
                sum_tree[i] = v

            # test set
            for i in range(size):
                self.assertEqual(sum_tree[i], x[i])

            # test sum
            self.assertEqual(sum_tree.sum(), x.sum())

            # test get_prefix_sum_idx
            self.run_test_get_prefix_sum_idx(x, sum_tree)


class TestMinTree(unittest.TestCase):
    def test_min_tree(self):

        for size in range(2, 16):
            x = np.random.choice(50, size=size)
            x += 1  # ensure positive
            min_tree = MinTree(size)

            for i, v in enumerate(x):
                min_tree[i] = v

            # test set
            for i in range(size):
                self.assertEqual(min_tree[i], x[i])

            # test min
            self.assertEqual(min_tree.min(), x.min())

    def test_min_tree_after_multiple_sets(self):

        for size in range(2, 16):
            x = np.random.choice(50, size=size)
            x += 1  # ensure positive
            min_tree = MinTree(size)

            for i, v in enumerate(x):
                min_tree[i] = v

            x = np.random.choice(50, size=size) + 1
            for i, v in enumerate(x):
                min_tree[i] = v

            # test set
            for i in range(size):
                self.assertEqual(min_tree[i], x[i])

            # test min
            self.assertEqual(min_tree.min(), x.min())


if __name__ == "__main__":
    unittest.main()
