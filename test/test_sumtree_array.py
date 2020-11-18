import unittest
import numpy as np
from starr import SumTreeArray


class TestSumTreeArray(unittest.TestCase):
    def test_array_creation_from_shape_inputs(self):
        starr = SumTreeArray(2)
        self.assertTrue(isinstance(starr, SumTreeArray))
        self.assertEqual(starr.size, 2)
        self.assertEqual(starr.min(), 0)
        self.assertEqual(starr.max(), 0)
        self.assertEqual(starr.dtype, float)
        self.assertFalse(starr.flags["WRITEABLE"])

    def test_array_creation_from_shape_inputs_and_dtype(self):
        starr = SumTreeArray(2, dtype=int)
        self.assertTrue(isinstance(starr, SumTreeArray))
        self.assertEqual(starr.dtype, int)
        self.assertFalse(starr.flags["WRITEABLE"])

    def test_array_creation_from_nd_array_input(self):
        input_array = np.array([0, 1]).astype("int32")
        starr = SumTreeArray(input_array)
        # SumTreeArray creates a new base array, and thus changes
        # to input_array should not be reflected in starr
        input_array[0] = 99
        self.assertTrue(isinstance(starr, SumTreeArray))
        self.assertEqual(starr[0], 0)
        self.assertEqual(starr[1], 1)
        self.assertEqual(starr._sumtree[1], 1)
        self.assertEqual(starr.dtype, np.dtype("int32"))
        self.assertFalse(starr.flags["WRITEABLE"])

    def test_array_creation_from_nd_array_input_and_dtype(self):
        input_array = np.array([0, 1]).astype("int32")
        starr = SumTreeArray(input_array, dtype="int64")
        # SumTreeArray creates a new base array, and thus changes
        # to input_array should not be reflected in starr
        input_array[0] = 99
        self.assertTrue(isinstance(starr, SumTreeArray))
        self.assertEqual(starr[0], 0)
        self.assertEqual(starr[1], 1)
        self.assertEqual(starr._sumtree[1], 1)
        self.assertEqual(starr.dtype, np.dtype("int64"))
        self.assertFalse(starr.flags["WRITEABLE"])

    def test_array_creation_from_starr_input(self):
        input_starr = SumTreeArray(np.array([0, 1]).astype("int32"))
        starr = SumTreeArray(input_starr)
        # SumTreeArray returns the input, and thus changes
        # to input_starr should be reflected in starr
        self.assertEqual(id(input_starr), id(starr))
        input_starr[0] = 99
        self.assertEqual(starr[0], 99)
        self.assertEqual(starr[1], 1)
        self.assertEqual(starr._sumtree[1], 100)
        self.assertEqual(starr.dtype, np.dtype("int32"))
        self.assertFalse(starr.flags["WRITEABLE"])

    def test_array_creation_from_starr_input_and_dtype(self):
        input_starr = SumTreeArray(np.array([0, 1]).astype("int32"))
        starr = SumTreeArray(input_starr, dtype="int64")
        # SumTreeArray creates a new underlying prefix sum tree, because the type signature is different,
        # and thus changes to input_starr should not be reflected in starr
        input_starr[0] = 99
        self.assertTrue(isinstance(starr, SumTreeArray))
        self.assertEqual(starr[0], 0)
        self.assertEqual(starr[1], 1)
        self.assertEqual(starr._sumtree[1], 1)
        self.assertEqual(starr.dtype, np.dtype("int64"))
        self.assertFalse(starr.flags["WRITEABLE"])

    def test_array_creation_from_view(self):
        x = np.array([1, 2, 3])
        y = x.view(SumTreeArray)
        self.assertTrue(isinstance(y, SumTreeArray))
        self.assertTrue(np.shares_memory(x, y))
        # SumTreeArray should be read-only, though base array is still writeable
        self.assertTrue(x.flags["WRITEABLE"])
        self.assertFalse(y.flags["WRITEABLE"])
        # sumtree should be up-to-date
        self.assertEqual(y._sumtree[1], 6)

    def test_invalid_array_creation(self):
        with self.assertRaises(ValueError):
            SumTreeArray(0)
        with self.assertRaises(ValueError):
            SumTreeArray(1)
        with self.assertRaises(ValueError):
            SumTreeArray(np.array(1))
        with self.assertRaises(ValueError):
            SumTreeArray(np.array([1]))

    def test_astype(self):
        starr = SumTreeArray(np.array([0, 1]).astype("int32"))
        output = starr.astype("int32")
        self.assertTrue(isinstance(output, np.ndarray))
        self.assertFalse(isinstance(output, SumTreeArray))
        self.assertFalse(np.shares_memory(starr, output))

    def test_view(self):
        starr = SumTreeArray(np.array([0, 1]).astype("int32"))
        output = starr.view(np.ndarray)
        self.assertTrue(isinstance(output, np.ndarray))
        self.assertFalse(isinstance(output, SumTreeArray))
        self.assertTrue(np.shares_memory(starr, output))
        self.assertFalse(output.flags["WRITEABLE"])

    def test_fill(self):
        starr = SumTreeArray(np.array([0, 1]).astype("int32"))
        starr.fill(99)
        self.assertEqual(starr[0], 99)
        self.assertEqual(starr[1], 99)
        self.assertEqual(starr._sumtree[1], 99 + 99)

    def test_reshape(self):
        x1 = SumTreeArray(np.array([0, 1, 2, 3]).astype("int32"))
        x2 = x1.reshape((2, 2))
        # when reshaping, the underlying data, index, and sumtree objects are shared
        self.assertNotEqual(id(x1), id(x2))
        self.assertEqual(id(x1._flat_base), id(x2._flat_base))
        self.assertEqual(id(x1._sumtree), id(x2._sumtree))
        self.assertEqual(id(x1._indices.base), id(x2._indices.base))
        # and thus changes in x1 will be reflected in x2, and vice versa
        x1[0] = 10
        self.assertEqual(x1[0], 10)
        self.assertEqual(x1._sumtree[1], 16)
        self.assertEqual(x2[0, 0], 10)
        self.assertEqual(x2._sumtree[1], 16)
        x2[1] = 20
        self.assertEqual(x1[0], 10)
        self.assertEqual(x1[1], 1)
        self.assertEqual(x1[2], 20)
        self.assertEqual(x1[3], 20)
        self.assertEqual(x1._sumtree[1], 51)
        self.assertEqual(x2[0, 0], 10)
        self.assertEqual(x2[0, 1], 1)
        self.assertEqual(x2[1, 0], 20)
        self.assertEqual(x2[1, 1], 20)
        self.assertEqual(x2._sumtree[1], 51)

    def test_ufunc(self):
        x = SumTreeArray(np.array([0, 1, 2, 3]).astype("int32"))
        x2 = x + 1
        # when a transformation is applied to a SumTreeArray object, it is assumed that
        # we do not want a new SumTreeArray object (which could result in a large
        # number of unwanted prefix sum tree updates)...and thus the transformation
        # is applied to the underlying array object, and an NDArray is returned
        self.assertFalse(isinstance(x2, SumTreeArray))

        # underlying x and sumtree is unchanged
        self.assertEqual(x2[0], 1)
        self.assertEqual(x[0], 0)
        self.assertEqual(x._sumtree[1], 6)

        # in-place assignment operators update the sumtree
        x3 = x
        x3 += 10
        self.assertTrue(isinstance(x3, SumTreeArray))
        self.assertEqual(x3[0], 10)
        self.assertEqual(x[0], 10)
        self.assertEqual(x._sumtree[1], 46)
        self.assertEqual(x3._sumtree[1], 46)

    def test_set(self):
        # [[0,1],
        #  [2,3],
        #  [4,5]]
        x = SumTreeArray(np.array([0, 1, 2, 3, 4, 5]).astype("int32").reshape((3, 2)))
        # normal array indexing
        x[1] = 10
        self.assertEqual(x[0, 0], 0)
        self.assertEqual(x[0, 1], 1)
        self.assertEqual(x[1, 0], 10)
        self.assertEqual(x[1, 1], 10)
        self.assertEqual(x[2, 0], 4)
        self.assertEqual(x[2, 1], 5)
        self.assertEqual(x._sumtree[1], 30)
        # array of indexes
        x[np.array([0, 1]), np.array([1, 0])] = 20
        self.assertEqual(x[0, 0], 0)
        self.assertEqual(x[0, 1], 20)
        self.assertEqual(x[1, 0], 20)
        self.assertEqual(x[1, 1], 10)
        self.assertEqual(x[2, 0], 4)
        self.assertEqual(x[2, 1], 5)
        self.assertEqual(x._sumtree[1], 59)
        # boolean indexing
        x[np.array([True, False, True])] = 30
        self.assertEqual(x[0, 0], 30)
        self.assertEqual(x[0, 1], 30)
        self.assertEqual(x[1, 0], 20)
        self.assertEqual(x[1, 1], 10)
        self.assertEqual(x[2, 0], 30)
        self.assertEqual(x[2, 1], 30)
        self.assertEqual(x._sumtree[1], 150)

    def test_get(self):
        # [[0,1],
        #  [2,3],
        #  [4,5]]
        x = SumTreeArray(np.array([0, 1, 2, 3, 4, 5]).astype("int32").reshape((3, 2)))
        # normal array indexing (slice shares memory)
        y = x[1]
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertTrue(np.shares_memory(x, y))
        self.assertFalse(y.flags["WRITEABLE"])
        self.assertEqual(y[0], 2)
        self.assertEqual(y[1], 3)
        self.assertEqual(y.size, 2)
        # normal array indexing (slice shares memory)
        y = x[1:3, 1:]
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertTrue(np.shares_memory(x, y))
        self.assertFalse(y.flags["WRITEABLE"])
        self.assertEqual(y[0], 3)
        self.assertEqual(y[1], 5)
        self.assertEqual(y.size, 2)
        # array of indexes
        y = x[np.array([0, 1]), np.array([1, 0])]
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertFalse(np.shares_memory(x, y))
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 2)
        self.assertEqual(y.size, 2)
        # boolean indexing
        y = x[np.array([True, False, True])]
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertFalse(np.shares_memory(x, y))
        self.assertEqual(y[0, 0], 0)
        self.assertEqual(y[0, 1], 1)
        self.assertEqual(y[1, 0], 4)
        self.assertEqual(y[1, 1], 5)
        self.assertEqual(y.size, 4)

    def test_get_prefix_sum_id(self):
        # This validates pipes are setup correctly.
        # Does not test full functionality of get_prefix_sum_idx, since that is
        # tested elsewhere.
        x = SumTreeArray(np.array([1, 2, 3, 4]).astype("int32"))
        # single element
        y = x.get_prefix_sum_id(1.5)
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y[0], 1)
        self.assertEqual(y.size, 1)
        # array of sums
        y = x.get_prefix_sum_id([[0.5, 1.5], [3.5, 6.5]], flatten_indices=False)
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y[0, 0], 0)
        self.assertEqual(y[0, 1], 1)
        self.assertEqual(y[1, 0], 2)
        self.assertEqual(y[1, 1], 3)
        self.assertEqual(y.size, 4)
        # array of sums (doesn't change when flatten_indices=True)
        y = x.get_prefix_sum_id([[0.5, 1.5], [3.5, 6.5]], flatten_indices=True)
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y[0, 0], 0)
        self.assertEqual(y[0, 1], 1)
        self.assertEqual(y[1, 0], 2)
        self.assertEqual(y[1, 1], 3)
        self.assertEqual(y.size, 4)
        # works for 2d SumTreeArray
        x = SumTreeArray(np.array([[1, 2], [3, 4]]).astype("int32"))
        # single element
        y = x.get_prefix_sum_id(1.5, flatten_indices=False)
        self.assertTrue(isinstance(y, tuple))
        self.assertEqual(len(y), 2)
        self.assertTrue(isinstance(y[0], np.ndarray))
        self.assertTrue(isinstance(y[1], np.ndarray))
        self.assertFalse(isinstance(y[0], SumTreeArray))
        self.assertFalse(isinstance(y[1], SumTreeArray))
        self.assertEqual(y[0][0], 0)
        self.assertEqual(y[0].size, 1)
        self.assertEqual(y[1][0], 1)
        self.assertEqual(y[1].size, 1)
        # single element
        y = x.get_prefix_sum_id(1.5, flatten_indices=True)
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y[0], 1)
        self.assertEqual(y.size, 1)
        # flat array of sums
        y = x.get_prefix_sum_id([0.5, 1.5, 3.5, 6.5], flatten_indices=True)
        self.assertEqual(y[0], 0)
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y[1], 1)
        self.assertEqual(y[2], 2)
        self.assertEqual(y[3], 3)
        self.assertEqual(y.size, 4)
        # 2d array of sums
        y = x.get_prefix_sum_id([[0.5, 1.5], [3.5, 6.5]], flatten_indices=True)
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y[0, 0], 0)
        self.assertEqual(y[0, 1], 1)
        self.assertEqual(y[1, 0], 2)
        self.assertEqual(y[1, 1], 3)
        self.assertEqual(y.size, 4)
        # returns tuple of flat array of indices
        y = x.get_prefix_sum_id([0.5, 1.5, 3.5, 6.5], flatten_indices=False)
        self.assertTrue(isinstance(y, tuple))
        self.assertEqual(len(y), 2)
        self.assertTrue(isinstance(y[0], np.ndarray))
        self.assertTrue(isinstance(y[1], np.ndarray))
        self.assertFalse(isinstance(y[0], SumTreeArray))
        self.assertFalse(isinstance(y[1], SumTreeArray))
        self.assertEqual(y[0][0], 0)
        self.assertEqual(y[0][1], 0)
        self.assertEqual(y[0][2], 1)
        self.assertEqual(y[0][3], 1)
        self.assertEqual(y[0].size, 4)
        self.assertEqual(y[1][0], 0)
        self.assertEqual(y[1][1], 1)
        self.assertEqual(y[1][2], 0)
        self.assertEqual(y[1][3], 1)
        self.assertEqual(y[1].size, 4)
        # returns tuple of 2d array of indices
        y = x.get_prefix_sum_id([[0.5, 1.5], [3.5, 6.5]], flatten_indices=False)
        self.assertTrue(isinstance(y, tuple))
        self.assertEqual(len(y), 2)
        self.assertTrue(isinstance(y[0], np.ndarray))
        self.assertTrue(isinstance(y[1], np.ndarray))
        self.assertFalse(isinstance(y[0], SumTreeArray))
        self.assertFalse(isinstance(y[1], SumTreeArray))
        self.assertEqual(y[0][0, 0], 0)
        self.assertEqual(y[0][0, 1], 0)
        self.assertEqual(y[0][1, 0], 1)
        self.assertEqual(y[0][1, 1], 1)
        self.assertEqual(y[0].size, 4)
        self.assertEqual(y[1][0, 0], 0)
        self.assertEqual(y[1][0, 1], 1)
        self.assertEqual(y[1][1, 0], 0)
        self.assertEqual(y[1][1, 1], 1)
        self.assertEqual(y[1].size, 4)

    def test_sample(self):
        # flat index
        x = SumTreeArray(np.array([1, 2, 3, 4]).astype("int32"))
        # single sample
        idx = x.sample(flatten_indices=False)
        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertEqual(idx.size, 1)
        self.assertEqual(x[idx].size, 1)
        # single sample
        idx = x.sample(flatten_indices=True)
        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertEqual(idx.size, 1)
        self.assertEqual(x[idx].size, 1)
        # multiple samples
        idx = x.sample(20, flatten_indices=False)
        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertEqual(x[idx].size, 20)
        # multiple samples and flatten_indices=True doesn't change result
        idx = x.sample(20, flatten_indices=True)
        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertEqual(x[idx].size, 20)
        # 2d index
        x = SumTreeArray(np.array([[1, 2], [3, 4]]).astype("int32"))
        # single sample
        idx = x.sample(flatten_indices=False)
        self.assertTrue(isinstance(idx, tuple))
        self.assertEqual(len(idx), 2)
        self.assertTrue(isinstance(idx[0], np.ndarray))
        self.assertTrue(isinstance(idx[1], np.ndarray))
        self.assertEqual(x[idx].size, 1)
        # multiple samples
        idx = x.sample(20, flatten_indices=False)
        self.assertTrue(isinstance(idx, tuple))
        self.assertEqual(len(idx), 2)
        self.assertTrue(isinstance(idx[0], np.ndarray))
        self.assertTrue(isinstance(idx[1], np.ndarray))
        self.assertEqual(x[idx].size, 20)
        # multiple samples and flatten_indices=True returns flattened indices
        idx = x.sample(20, flatten_indices=True)
        self.assertTrue(isinstance(idx, np.ndarray))
        self.assertEqual(len(idx), 20)
        self.assertEqual(idx.ndim, 1)

    def test_sum(self):
        x = SumTreeArray(np.array([1, 2, 3, 4]).astype("int32"))
        self.assertEqual(x.sum(), 10)
        x = SumTreeArray(np.array([[1, 2], [3, 4]]).astype("int32"))
        self.assertEqual(x.sum(), 10)
        y = x.sum(keepdims=True)
        self.assertEqual(y.ravel()[0], 10)
        self.assertEqual(y.size, 1)
        self.assertEqual(y.ndim, x.ndim)
        z = x.sum(axis=1)
        self.assertEqual(z[0], 3)
        self.assertEqual(z[1], 7)
        self.assertEqual(z.size, 2)

    def test_sum_with_args(self):
        v = np.random.choice(100, size=(100, 100))
        x = SumTreeArray(v)
        self.assertFalse(isinstance(x.sum(axis=1), SumTreeArray))
        self.assertTrue(isinstance(x.sum(axis=1), np.ndarray))
        self.assertFalse(isinstance(x.sum(axis=1, keepdims=True), SumTreeArray))
        self.assertTrue(isinstance(x.sum(axis=1, keepdims=True), np.ndarray))
        self.assertEqual(0, np.abs(x.sum(axis=1) - v.sum(axis=1)).max())
        self.assertEqual(0, np.abs(x.sum(axis=0) - v.sum(axis=0)).max())
        self.assertEqual(
            0, np.abs(x.sum(axis=1, keepdims=True) - v.sum(axis=1, keepdims=True)).max()
        )
        self.assertEqual(
            0, np.abs(x.sum(axis=0, keepdims=True) - v.sum(axis=0, keepdims=True)).max()
        )
        self.assertEqual(0, np.abs(x.sum() - v.sum()).max())

    def test_parse_axis_arg(self):
        x = SumTreeArray((2, 3, 4))

        def check_normal_inputs(axis, expected):
            output = x._parse_axis_arg(axis)
            self.assertTrue(isinstance(output, np.ndarray))
            self.assertEqual(output.ndim, 1)
            self.assertEqual(len(output), len(expected))
            self.assertTrue(np.issubdtype(output.dtype, np.integer))
            self.assertTrue(np.all(output == expected))

        self.assertEqual(x._parse_axis_arg(None), None)
        check_normal_inputs(0, np.array([0], dtype=int))
        check_normal_inputs((0, 1), np.array([0, 1], dtype=int))
        check_normal_inputs((1, -1), np.array([1, 2], dtype=int))
        check_normal_inputs(-1, np.array([2], dtype=int))
        check_normal_inputs(-2, np.array([1], dtype=int))

        with self.assertRaises(IndexError):
            x._parse_axis_arg((1, 1))
        with self.assertRaises(IndexError):
            x._parse_axis_arg(3)

    def test_invalid_entry_error(self):
        with self.assertRaises(ValueError):
            SumTreeArray(np.array([-1, 0, 1]).astype("int32"))
        with self.assertRaises(ValueError):
            x = SumTreeArray(np.array([1, 2, 3, 4]).astype("int32"))
            x[:2] = -1

    def test_sample_validation(self):
        with self.assertRaises(ValueError):
            x = SumTreeArray(10)
            x.sample(10)

    def test_sumtree(self):
        x = SumTreeArray(np.arange(4))
        y = x.sumtree()
        y[0] = 100
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(y[0], 100)
        self.assertEqual(x._sumtree[0], 0)

    def test_rebuild_sumtree(self):
        x = SumTreeArray(np.arange(4))
        x._sumtree.fill(0)
        self.assertEqual(x._sumtree[1], 0)
        x._rebuild_sumtree()
        self.assertEqual(x._sumtree[1], 6)

    def test_enable_writes(self):
        x = SumTreeArray(np.arange(4))
        self.assertFalse(x.flags["WRITEABLE"])
        self.assertTrue(x.base.flags["WRITEABLE"])
        x._enable_writes(True)
        self.assertTrue(x.flags["WRITEABLE"])
        self.assertTrue(x.base.flags["WRITEABLE"])
        x._enable_writes(False)
        self.assertFalse(x.flags["WRITEABLE"])
        self.assertTrue(x.base.flags["WRITEABLE"])

    def test_copy(self):
        x = SumTreeArray(np.arange(4))
        y = x.copy()
        self.assertTrue(isinstance(y, SumTreeArray))
        self.assertFalse(np.shares_memory(x, y))
        self.assertFalse(np.shares_memory(x._flat_base, y._flat_base))
        self.assertFalse(np.shares_memory(x._sumtree, y._sumtree))
        self.assertFalse(np.shares_memory(x._indices, y._indices))
        self.assertEqual(y.sum(), x.sum())
        self.assertEqual(np.max(np.abs(x - y)), 0)
        self.assertEqual(np.max(np.abs(x._flat_base - y._flat_base)), 0)
        self.assertEqual(np.max(np.abs(x._sumtree - y._sumtree)), 0)
        self.assertEqual(np.max(np.abs(x._indices - y._indices)), 0)

    def test_choose(self):
        x1 = np.array([0, 1])
        x2 = SumTreeArray(x1)
        y1 = x1.choose([np.array([0, 1]), np.array([2, 3])])
        y2 = x2.choose([np.array([0, 1]), np.array([2, 3])])
        self.assertFalse(isinstance(y2, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)

    def test_diagonal(self):
        x = SumTreeArray(np.array([[0, 1], [2, 3]]))
        y = x.diagonal()
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertEqual(np.max(np.abs(y - np.array([0, 3]))), 0)

    def test_dot(self):
        x1 = SumTreeArray(np.array([[0, 1], [2, 3]]))
        y1 = x1.dot(x1)
        x2 = np.array([[0, 1], [2, 3]])
        y2 = x2.dot(x2)
        self.assertFalse(isinstance(y1, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)

    def test_flatten(self):
        x = SumTreeArray(np.array([0, 1]))
        y = x.flatten()
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertFalse(np.shares_memory(x, y))

    def test_imag(self):
        x = SumTreeArray(np.array([0, 1]))
        y = x.imag
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertFalse(np.shares_memory(x, y))

    def test_mean(self):
        x1 = np.array([[0, 1, 2], [3, 4, 5]])
        x2 = SumTreeArray(x1)
        self.assertEqual(x1.mean(), x2.mean())
        self.assertEqual(np.max(np.abs(x1.mean(axis=0) - x2.mean(axis=0))), 0)
        self.assertEqual(np.max(np.abs(x1.mean(axis=1) - x2.mean(axis=1))), 0)

    def test_newbyteorder(self):
        x = SumTreeArray(np.array([2, 3, 1, 0])).newbyteorder()
        y = np.array([2, 3, 1, 0]).newbyteorder()
        self.assertEqual(np.max(np.abs(x - y)), 0)

    def test_partition(self):
        x = SumTreeArray(np.array([2, 3, 1, 0]))
        y = np.array([2, 3, 1, 0])
        x.partition(2)
        y.partition(2)
        self.assertEqual(np.max(np.abs(x - y)), 0)

    def test_put(self):
        # [[0,1],
        #  [2,3],
        #  [4,5]]
        x = SumTreeArray(np.array([0, 1, 2, 3, 4, 5]).astype("int32").reshape((3, 2)))
        # normal array indexing
        x.put(1, 10)
        self.assertEqual(x[0, 0], 0)
        self.assertEqual(x[0, 1], 1)
        self.assertEqual(x[1, 0], 10)
        self.assertEqual(x[1, 1], 10)
        self.assertEqual(x[2, 0], 4)
        self.assertEqual(x[2, 1], 5)
        self.assertEqual(x._sumtree[1], 30)
        # array of indexes
        x.put((np.array([0, 1]), np.array([1, 0])), 20)
        self.assertEqual(x[0, 0], 0)
        self.assertEqual(x[0, 1], 20)
        self.assertEqual(x[1, 0], 20)
        self.assertEqual(x[1, 1], 10)
        self.assertEqual(x[2, 0], 4)
        self.assertEqual(x[2, 1], 5)
        self.assertEqual(x._sumtree[1], 59)
        # boolean indexing
        x.put(np.array([True, False, True]), 30)
        self.assertEqual(x[0, 0], 30)
        self.assertEqual(x[0, 1], 30)
        self.assertEqual(x[1, 0], 20)
        self.assertEqual(x[1, 1], 10)
        self.assertEqual(x[2, 0], 30)
        self.assertEqual(x[2, 1], 30)
        self.assertEqual(x._sumtree[1], 150)

    def test_real(self):
        x = SumTreeArray(np.array([0, 1]))
        y = x.real
        self.assertFalse(isinstance(y, SumTreeArray))
        self.assertTrue(np.shares_memory(x, y))

    def test_repeat(self):
        x1 = SumTreeArray(np.array([[0, 1], [2, 3]]))
        x2 = np.array([[0, 1], [2, 3]])
        y1 = x1.repeat(5)
        y2 = x2.repeat(5)
        self.assertFalse(isinstance(y1, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)

    def test_round(self):
        x1 = np.array([[0.1, 1.2], [2.5, 3.6]])
        x2 = SumTreeArray(x1)
        y1 = x1.round(0)
        y2 = x2.round(0)
        self.assertFalse(isinstance(y2, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)

    def test_sort(self):
        x = SumTreeArray(np.array([2, 3, 1, 0]))
        x.sort()
        self.assertEqual(np.max(np.abs(x - np.arange(4))), 0)

    def test_squeeze(self):
        x = SumTreeArray(np.array([2, 3, 1, 0]).reshape([1, 1, 2, 1, 2, 1, 1]))
        y = x.squeeze()
        self.assertTrue(isinstance(y, SumTreeArray))
        self.assertTrue(np.shares_memory(x, y))
        self.assertTrue(np.shares_memory(x._flat_base, y._flat_base))
        self.assertTrue(np.shares_memory(x._sumtree, y._sumtree))
        self.assertEqual(y.shape, (2, 2))

    def test_swapaxes(self):
        x1 = np.array([0, 1, 2, 3]).reshape(2, 2)
        x2 = SumTreeArray(x1)
        y1 = x1.swapaxes(0, 1)
        y2 = x2.swapaxes(0, 1)
        self.assertFalse(isinstance(y2, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)

    def test_T(self):
        x1 = np.array([0, 1, 2, 3]).reshape(2, 2)
        x2 = SumTreeArray(x1)
        y1 = x1.T
        y2 = x2.T
        self.assertFalse(isinstance(y2, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)

    def test_take(self):
        x1 = np.array([[0, 1], [2, 3]])
        x2 = SumTreeArray(x1)
        y1 = x1.take([0, 3])
        y2 = x2.take([0, 3])
        self.assertFalse(isinstance(y2, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)

    def test_trace(self):
        x1 = np.array([[0, 1], [2, 3]])
        x2 = SumTreeArray(x1)
        y1 = x1.trace()
        y2 = x2.trace()
        self.assertFalse(isinstance(y2, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)

    def test_transpose(self):
        x1 = np.array([0, 1, 2, 3]).reshape(2, 2)
        x2 = SumTreeArray(x1)
        y1 = x1.transpose()
        y2 = x2.transpose()
        self.assertFalse(isinstance(y2, SumTreeArray))
        self.assertEqual(np.max(np.abs(y1 - y2)), 0)


if __name__ == "__main__":
    unittest.main()
