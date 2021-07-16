![Build](https://github.com/justinmaojones/starr/workflows/Build/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://github.com/justinmaojones/starr/docs/badges/python.svg)
![Coverage](https://github.com/justinmaojones/starr/docs/badges/coverage.svg)

# STArr

Fast sum tree ops in Cython for NumPy arrays.  Inspired by [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

## Installation

```
pip install starr 
```

## Quickstart 

Initialize a `SumTreeArray`, a subclass of `numpy.ndarray`
```python
>>> from starr import SumTreeArray
>>> sumtree_array = SumTreeArray(4, dtype='float32')
>>> sumtree_array
SumTreeArray([0., 0., 0., 0.], dtype=float32)
```

Or build one from an existing n-dimensional `ndarray` 
```python
>>> import numpy as np
>>> sumtree_array_2d = SumTreeArray(np.array([[1,2,3],[4,5,6]], dtype='int32'))
>>> sumtree_array_2d
SumTreeArray([[1, 2, 3],
              [4, 5, 6]], dtype=int32)
```

Set values like you normally would
```python
>>> sumtree_array[0] = 1
>>> sumtree_array[1:2] = [2]
>>> sumtree_array[np.array([False,False,True,False])] = 3
>>> sumtree_array[-1] = 4
>>> sumtree_array
SumTreeArray([1., 2., 3., 4.], dtype=float32)
```

A `SumTreeArray` maintains an internal sum tree, which can be used for fast sampling and sum ops.
```python
>>> sumtree_array.sumtree()
array([ 0., 10.,  3.,  7.], dtype=float32)
```

Sample indices (efficiently), where each element is the unnormalized probability of being sampled
```python
>>> sumtree_array.sample(10)
array([2, 3, 3, 3, 3, 1, 2, 2, 2, 0], dtype=int32)

>>> # probability of being sampled
>>> sumtree_array / sumtree_array.sum() 
array([0.1, 0.2, 0.3, 0.4], dtype=float32)

>>> # sampled proportions
>>> (sumtree_array.sample(1000)[None] == np.arange(4)[:,None]).mean(axis=1) 
array([0.10057, 0.19919, 0.29983, 0.40041])
```

You can also sample indices from an n-dimensional `SumTreeArray`
```python
>>> sumtree_array_2d.sample(4)
(array([1, 1, 0, 0]), array([0, 1, 1, 2]))
```

Use the array's `sum` method to use the sumtree to calculate sums (quickly)
```python
>>> sumtree_array.sum()
10.0
```

## Memory

Arithmetic operations return `ndarray` (to avoid expensive tree initialization) 
```python
>>> sumtree_array * 2
array([ 2., 4., 6., 8.], dtype=float32)
```

This is true for get operations as well
```python
>>> sumtree_array[1:3]
array([2., 3.], dtype=float32)

>>> sumtree_array[:]
array([1., 2., 3., 4.], dtype=float32)
```

However, in-place operations update `SumTreeArray` 
```python
>>> sumtree_array_in_place_op = SumTreeArray(np.array([2,4,6,8]),dtype='float32')
>>> sumtree_array_in_place_op += 1
>>> sumtree_array_in_place_op 
SumTreeArray([3., 5., 7., 9.], dtype=float32)
```

## Performance

See [latest benchmarks](starr/experimental/README.md).

Sampling indices is faster than normal sampling methods in `numpy`
```python
>>> x = SumTreeArray(np.ones(int(1e6)))
>>> %timeit x.sample(100)
55.2 µs ± 6.17 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> y = np.ones(int(1e6))
>>> %timeit np.random.choice(len(y),size=100,p=y/y.sum())
10.8 ms ± 697 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

For large arrays, sum operations over C-contiguous blocks of memory are faster than `ndarray`, because of the sum tree:
```python
>>> x = SumTreeArray(np.ones((1000,1000)))
>>> %timeit x.sum()
428 ns ± 10.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

>>> y = np.ones((1000,1000))
>>> %timeit y.sum()
272 µs ± 51.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

>>> %timeit x.sum(axis=1)
118 µs ± 2.2 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> %timeit y.sum(axis=1)
276 µs ± 68.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

Sum operations over non C-contiguous blocks of memory (e.g. along the first axis of a 2d array) are slower: 
```python
>>> %timeit x.sum(axis=0)
367 µs ± 28 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

>>> %timeit y.sum(axis=0)
303 µs ± 6.97 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

Set operations are much slower in `SumTreeArray` than in `ndarray`, because each set operation updates the tree, but that's okay when using `SumTreeArray` for applications that rely heavily on sampling and sum operations, such as prioritzed experience replay!  In the example below, updating and sampling with `SumTreeArray` is 150x faster than with `ndarray`, even though the update operation alone in `ndarray` is 26x faster than `SumTreeArray`!
```python
>>> x = SumTreeArray(np.ones(int(1e6)))

>>> # set + sample 
>>> %timeit x[-10:] = 2; x.sample(100)
71.4 µs ± 3.71 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> y = np.ones(int(1e6))
>>> y_sum = y.sum() # let's assume we keep track of this efficiently

>>> # set + sample 
>>> %timeit y[-10:] = 2; np.random.choice(len(y),size=100,p=y/y_sum)
10.7 ms ± 752 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
