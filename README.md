# CythonSumTree 

Fast sum segment trees in Cython for Numpy arrays.  Inspired by the [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) paper.

## Installation

```
pip install cython_sum_tree
```

## Quickstart 

Initialize a `PrefixSumTree`:

```python
>>> from prefix_sum_tree import PrefixSumTree
>>> sum_tree = PrefixSumTree(4,dtype='float32')

>>> sum_tree
PrefixSumTree([0., 0., 0., 0.], dtype=float32)
```

or build one from an existing n-dimensional numpy array
```python
>>> import numpy as np
>>> sum_tree_from_2d_array = PrefixSumTree(np.array([[1,2,3],[4,5,6]],dtype='int32'))

>>> sum_tree_from_2d_array
PrefixSumTree([[1, 2, 3],
               [4, 5, 6]], dtype=int32)
```

set values like you normally would with numpy 
```python
>>> sum_tree[0] = 1
>>> sum_tree[1:2] = [2]
>>> sum_tree[np.array([False,False,True,False])] = 3
>>> sum_tree[-1] = 4

>>> sum_tree
PrefixSumTree([1., 2., 3., 4.], dtype=float32)
```

sample indices (quickly), with each element containing the unnormalized probability of being sampled
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

you can also sample indices from an n-dimensional `PrefixSumTree`
```python
>>> sum_tree_from_2d_array.sample(4)
(array([1, 1, 0, 0]), array([0, 1, 1, 2]))
```

for large arrays, sum operations over C-contiguous blocks of memory are faster (because of the sum tree):
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

sum operations over non C-contiguous blocks of memory (e.g. along the first axis of a 2d array) are slower: 
```python
>>> %timeit x.sum(axis=0)
367 µs ± 28 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

>>> %timeit y.sum(axis=0)
303 µs ± 6.97 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```
