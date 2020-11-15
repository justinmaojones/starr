![Build](https://github.com/justinmaojones/cy_prefix_sum_tree/workflows/Build/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](docs/badges/python.svg)
![Coverage](docs/badges/coverage.svg)

# CythonSumTree 

Fast sum segment trees in C (via Cython) for Numpy arrays in Python.  Inspired by [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

## Installation

```
pip install cython_sum_tree
```

## Quickstart 

Initialize a `PrefixSumTree`, which subclasses `numpy.ndarray`
```python
>>> from prefix_sum_tree import PrefixSumTree
>>> sum_tree = PrefixSumTree(4,dtype='float32')
>>> sum_tree
PrefixSumTree([0., 0., 0., 0.], dtype=float32)
```

Or build one from an existing n-dimensional `numpy.ndarray` 
```python
>>> import numpy as np
>>> sum_tree_from_2d_array = PrefixSumTree(np.array([[1,2,3],[4,5,6]],dtype='int32'))
>>> sum_tree_from_2d_array
PrefixSumTree([[1, 2, 3],
               [4, 5, 6]], dtype=int32)
```

Set values like a `numpy.ndarray`
```python
>>> sum_tree[0] = 1
>>> sum_tree[1:2] = [2]
>>> sum_tree[np.array([False,False,True,False])] = 3
>>> sum_tree[-1] = 4
>>> sum_tree
PrefixSumTree([1., 2., 3., 4.], dtype=float32)
```

A `PrefixSumTree` maintains a sum segment tree, which can be used for fast sum and sampling.
```python
>>> sum_tree.sumtree()
array([ 0., 10.,  3.,  7.], dtype=float32)
```

Arithmetic operations return a new `ndarray` (to avoid expensive tree initialization) 
```python
>>> sum_tree * 2
array([ 2., 4., 6., 8.], dtype=float32)
```

This is true for get operations as well
```python
>>> sum_tree[1:3]
array([2., 3.], dtype=float32)

>>> sum_tree[:]
array([1., 2., 3., 4.], dtype=float32)
```

However, in-place operations update `PrefixSumTree` 
```python
>>> sum_tree_in_place_op = PrefixSumTree(np.array([2,4,6,8]),dtype='float32')
>>> sum_tree_in_place_op += 1
>>> sum_tree_in_place_op 
PrefixSumTree([3., 5., 7., 9.], dtype=float32)
```

Sample indices (efficiently), with each element containing the unnormalized probability of being sampled
```python
>>> sum_tree.sample(10)
array([2, 3, 3, 3, 3, 1, 2, 2, 2, 0], dtype=int32)

>>> # probability of being sampled
>>> sum_tree / sum_tree.sum() 
array([0.1, 0.2, 0.3, 0.4], dtype=float32)

>>> # sampled proportions
>>> (sum_tree.sample(1000)[None] == np.arange(4)[:,None]).mean(axis=1) 
array([0.10057, 0.19919, 0.29983, 0.40041])
```

You can also sample indices from an n-dimensional `PrefixSumTree`
```python
>>> sum_tree_from_2d_array.sample(4)
(array([1, 1, 0, 0]), array([0, 1, 1, 2]))
```

## Performance

Sampling indices is faster than normal sampling methods in `numpy`
```python
>>> x = PrefixSumTree(np.ones(int(1e6)))
>>> %timeit x.sample(100)
55.2 µs ± 6.17 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> y = np.ones(int(1e6))
>>> %timeit np.random.choice(len(y),size=100,p=y/y.sum())
10.8 ms ± 697 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

For large arrays, sum operations over C-contiguous blocks of memory are faster than `ndarray`, because of the sum tree:
```python
>>> x = PrefixSumTree(np.ones((1000,1000)))
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

Set operations are much slower in `PrefixSumTree` than in `ndarray`, because each set operation updates the tree, but that's okay when using `PrefixSumTree` for applications that rely heavily on sampling and sum operations, such as prioritzed experience replay!  In the example below, updating and sampling with `PrefixSumTree` is 150x faster than with `ndarray`, even though the update operation alone in `ndarray` is 26x faster than `PrefixSumTree`!
```python
>>> x = PrefixSumTree(np.ones(int(1e6)))

>>> # set only 
>>> %timeit x[-10:] = 2
10.8 µs ± 525 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

>>> # set + sample 
>>> %timeit x[-10:] = 2; x.sample(100)
71.4 µs ± 3.71 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

>>> y = np.ones(int(1e6))
>>> y_sum = y.sum() # let's assume we keep track of this efficiently

>>> # set only 
>>> %timeit y[-10:] = 2
411 ns ± 28.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

>>> # set + sample 
>>> %timeit y[-10:] = 2; np.random.choice(len(y),size=100,p=y/y_sum)
10.7 ms ± 752 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
