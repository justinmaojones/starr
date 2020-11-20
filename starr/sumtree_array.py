import numpy as np

from ._cython import build_sumtree_from_array
from ._cython import get_prefix_sum_idx
from ._cython import strided_sum
from ._cython import update_sumtree


def _temporarily_enable_update(func):
    def wrapper(self, *args, **kwargs):
        self._enable_writes(True)
        try:
            output = func(self, *args, **kwargs)
        finally:
            self._enable_writes(False)
        return output

    return wrapper


class SumTreeArray(np.ndarray):
    """
    A subclass of an ``numpy.ndarray`` that maintains an internal sumtree
    for fast Categorical distribution sampling and fast sum operations,
    at the expense of slower write operations. Because ``SumTreeArray``
    is a subclass of ``numpy.ndarray``, it can be used just like a
    ``numpy.ndarray``. It also comes with three new methods: ``sample``,
    ``get_prefix_sum_id``, and ``sumtree``.

    All elements of ``SumTreeArray`` must be non-negative (explanation below).

    Parameters
    ----------
    shape_or_array : int, tuple of ints, array-like
        When an int or tuple of ints is provided, a ``SumTreeArray`` is created
        from an array of zeros of the given shape. When array-like, a
        ``SumTreeArray`` is created from the given array.
    dtype : data-type, optional
        When provided creates a ``SumTreeArray`` of the specified dtype. If not
        provided, and ``shape_or_array`` is a shape, then an array of zeros with
        dtype ``float`` is created.  Defaults to ``None``. All integer and floating
        point dtypes are supported. Because of the non-negativity requirement,
        complex dtypes are not supported.

    Notes
    -----
    Because the elements of a ``SumTreeArray`` represent an unnormalized
    Categorical probability distribution, we require that all elements of a
    ``SumTreeArray`` be non-negative.

    For the sake of the integrity of the sumtree, the memory of the array is
    carefully guarded. Elements of ``SumTreeArray`` should only be updated through
    the ``SumTreeArray`` API.  This ensures that the underlying sumtree correctly
    models the underlying array. For example, when creating a ``SumTreeArray``
    from ``another_array``, the new ``SumTreeArray`` uses a copy of ``another_array``.
    Thus, changes to ``another_array`` do not affect the ``SumTreeArray``. As another
    example, using ``SumTreeArray.view(np.ndarray)`` will return an object
    with read-only access to the underlying array ``SumTreeArray``.

    Similarly, when retrieving elements of a ``SumTreeArray`` through indexing,
    the returned object is always an ``np.ndarray`` that is a copy or read-only
    view of the underlying array in ``SumTreeArray``, and thus the integrity of the
    sumtree is protected. Another reason for doing this is that we always assume
    that we do not want to re-compute a new sumtree on top of the returned object,
    which could be unnecessarily expensive.

    References
    ----------
    [1] NumPy https://numpy.org

    Examples
    --------
    >>> SumTreeArray(4,dtype='int32')
    SumTreeArray([0, 0, 0, 0], dtype=int32)
    >>> SumTreeArray((2,2),dtype='int32')
    SumTreeArray([[0, 0],
                  [0, 0]], dtype=int32)
    >>> sum_tree = SumTreeArray(np.array([1,2,3,4],dtype='float32'))
    >>> sum_tree
    SumTreeArray([1., 2., 3., 4.], dtype=float32)
    >>> # set and get just like an ndarray
    >>> sum_tree[:2] = [2,1]
    >>> sum_tree
    SumTreeArray([2., 1., 3., 4.], dtype=float32)
    """

    def __new__(self, shape_or_array, dtype=None):

        if isinstance(shape_or_array, SumTreeArray):
            if dtype is None:
                return shape_or_array
            else:
                return SumTreeArray(shape_or_array.view(np.ndarray), dtype)

        elif isinstance(shape_or_array, np.ndarray):
            if shape_or_array.size <= 1:
                raise ValueError(
                    "input to SumTreeArray must have shape with at least 2 elements"
                )
            assert shape_or_array.size > 1
            dtype = shape_or_array.dtype if dtype is None else dtype
            array = shape_or_array.astype(dtype, copy=True)  # strictly copies
            return array.view(SumTreeArray)

        else:
            dtype = float if dtype is None else dtype
            array = np.zeros(shape_or_array, dtype=dtype)
            if array.size <= 1:
                raise ValueError(
                    "input to SumTreeArray must have shape with at least 2 elements"
                )
            return array.view(SumTreeArray)

    @_temporarily_enable_update
    def __array_finalize__(self, array):

        if not np.shares_memory(array, self):
            # note that we would have ended up here without overriding copy(...)
            raise NotImplementedError("input array and self must share memory")

        if isinstance(self.base, SumTreeArray) and self.base.dtype == self.dtype:
            # inherit the same base and sum tree
            if array.size != self.size:
                # we should never end up here
                raise NotImplementedError(
                    "input array and base SumTreeArray must have same number of elements"
                )
            self._flat_base = self.base._flat_base
            self._indices = self.base._indices.reshape(self.shape)
            self._sumtree = self.base._sumtree
        else:
            # initialize
            self._flat_base = array.view(np.ndarray).ravel()
            self._indices = np.arange(array.size, dtype=np.intp).reshape(array.shape)
            self._sumtree = np.zeros_like(self._flat_base)
            # sumtree needs to be initialize
            self._rebuild_sumtree()

    # When a transformation is applied to a SumTreeArray object, it is assumed that
    # we do not want a new SumTreeArray object (which could result in a large
    # number of unwanted prefix sum tree updates)...and thus the transformation
    # is applied to the underlying array object, and an NDArray is returned.
    # The exception to this rule is in-place operators, such as +=
    def __array_prepare__(self, out_arr, context=None):
        return out_arr.view(np.ndarray)

    def __array_wrap__(self, out_arr, context=None):
        return out_arr.view(np.ndarray)

    @_temporarily_enable_update
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if isinstance(x, SumTreeArray):
                inputs[i] = x.view(np.ndarray)
        inputs = tuple(inputs)
        if "out" in kwargs and kwargs["out"] is not None:
            out = kwargs["out"]
            if len(out) == 1 and out[0] is self:
                # this is an in-place update on self
                # proceed with update and then rebuild sumtree
                kwargs["out"] = (self.view(np.ndarray),)
                _ = super(SumTreeArray, self).__array_ufunc__(
                    ufunc, method, *inputs, **kwargs
                )
                self._rebuild_sumtree()
                return self
            else:
                raise NotImplementedError
        else:
            return super(SumTreeArray, self).__array_ufunc__(
                ufunc, method, *inputs, **kwargs
            )

    def _enable_writes(self, val):
        self.setflags(write=val)

    def __setitem__(self, idx, val):
        self.put(idx, val)

    def __getitem__(self, idx):
        return self.view(np.ndarray)[idx]

    def _rebuild_sumtree(self):
        build_sumtree_from_array(self._flat_base, self._sumtree)

    def astype(self, *args, **kwargs):
        return self.view(np.ndarray).astype(*args, **kwargs)

    def choose(self, *args, **kwargs):
        return self.view(np.ndarray).choose(*args, **kwargs)

    def copy(self, *args, **kwargs):
        return SumTreeArray(self.view(np.ndarray))

    def diagonal(self, *args, **kwargs):
        return self.view(np.ndarray).diagonal(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return self.view(np.ndarray).dot(*args, **kwargs)

    @_temporarily_enable_update
    def fill(self, *args, **kwargs):
        super(SumTreeArray, self).fill(*args, **kwargs)
        self._rebuild_sumtree()

    def flatten(self, *args, **kwargs):
        return self.view(np.ndarray).flatten(*args, **kwargs)

    @property
    def imag(self):
        return self.view(np.ndarray).imag

    def mean(self, *args, **kwargs):
        output = self.sum(*args, **kwargs)
        m = np.array(output).size
        n = self.size
        assert n % m == 0
        return output / float(n / m)

    def newbyteorder(self, *args, **kwargs):
        return self.view(np.ndarray).newbyteorder(*args, **kwargs)

    @_temporarily_enable_update
    def partition(self, *args, **kwargs):
        super(SumTreeArray, self).partition(*args, **kwargs)
        self._rebuild_sumtree()

    def put(self, indices, values):
        # TODO: there's probably a better way of building an index iterator
        indices = np.ascontiguousarray(self._indices[indices]).ravel()
        values = np.ascontiguousarray(values, dtype=self._flat_base.dtype).ravel()
        update_sumtree(indices, values, self._flat_base, self._sumtree)

    @property
    def real(self):
        return self.view(np.ndarray).real

    def repeat(self, *args, **kwargs):
        return self.view(np.ndarray).repeat(*args, **kwargs)

    def round(self, *args, **kwargs):
        return self.view(np.ndarray).round(*args, **kwargs)

    @_temporarily_enable_update
    def sort(self, *args, **kwargs):
        super(SumTreeArray, self).sort(*args, **kwargs)
        self._rebuild_sumtree()

    def squeeze(self):
        return self.reshape(self.view(np.ndarray).squeeze().shape)

    def sumtree(self):
        """
        Returns a copy of the sumtree.

        Returns
        -------
        sumtree : ndarray
            A copy of the (flat) sumtree object maintained by ``SumTreeArray``.

        Examples
        --------
        >>> sum_tree = SumTreeArray(np.array([1,2,3,4],dtype='float32'))
        >>> sum_tree
        SumTreeArray([1., 2., 3., 4.], dtype=float32)
        >>> sum_tree.sumtree()
        array([ 0., 10.,  3.,  7.], dtype=float32)
        >>> sum_tree_from_2d_array = SumTreeArray(np.array([[1,2],[3,4]],dtype='int32'))
        >>> sum_tree_from_2d_array
        SumTreeArray([[1., 2.],
                      [3., 4.]], dtype=float32)
        >>> sum_tree_from_2d_array.sumtree()
        array([ 0., 10.,  3.,  7.], dtype=float32)
        """
        return np.copy(self._sumtree)

    def swapaxes(self, *args, **kwargs):
        return self.view(np.ndarray).swapaxes(*args, **kwargs)

    @property
    def T(self):
        return self.view(np.ndarray).T

    def take(self, *args, **kwargs):
        return self.view(np.ndarray).take(*args, **kwargs)

    def trace(self, *args, **kwargs):
        return self.view(np.ndarray).trace(*args, **kwargs)

    def transpose(self, *args, **kwargs):
        return self.view(np.ndarray).transpose(*args, **kwargs)

    def get_prefix_sum_id(self, prefix_sum, flatten_indices=False):
        """
        Returns an array of indices of the same shape is the input array
        ``prefix_sum`` where each element ``i`` in the output is ``j``
        such that ``self.ravel()[:j+1] < prefix_sum[i]``.  In other words,
        the output array returns the index of the largest prefix sum of
        self that is less than the provided input.

        Parameters
        ----------
        prefix_sum : array-like
            An n-dimensional array of prefix sums.  Does not need to be
            of the same shape as ``self``.
        flatten_indices : bool, optional
            Defaults to ``False``.  When ``True``, returns the indices
            as if ``self`` is a 1d array (i.e. ``self.ravel()``)

        Returns
        -------
        prefix_sum_indices : ndarray, tuple(ndarray)
            A new array holding the result is returned containing the
            indices of the supplied prefix sums. The returned array
            has the same shape as the input array. When supplied with
            ``flatten_indices=False`` and ``self`` is an n-d array
            where ``n>1``, then a tuple of ``n`` arrays is returned
            where each element of the tuple corresponds to each dimension
            of the ``SumTreeArray``.

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        Examples
        --------
        >>> sum_tree = SumTreeArray(np.array([1,2,3,4],dtype='float32'))
        >>> sum_tree
        SumTreeArray([1., 2., 3., 4.], dtype=float32)
        >>> sum_tree.get_prefix_sum_id(np.arange(10))
        array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=int32)
        >>> sum_tree.get_prefix_sum_id(np.arange(10).reshape(2,5))
        array([[0, 1, 1, 2, 2],
               [2, 3, 3, 3, 3]], dtype=int32)
        >>> sum_tree_from_2d_array = SumTreeArray(np.array([[1,2],[3,4]],dtype='int32'))
        >>> sum_tree_from_2d_array
        SumTreeArray([[1., 2.],
                       [3., 4.]], dtype=float32)
        >>> # output is "flattened" and thus will be the same as ``sum_tree.get_prefix_sum_id`` above
        >>> sum_tree_from_2d_array.get_prefix_sum_id(np.arange(10))
        array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=int32)
        >>> sum_tree_from_2d_array.get_prefix_sum_id(np.arange(10).reshape(2,5))
        array([[0, 1, 1, 2, 2],
               [2, 3, 3, 3, 3]], dtype=int32)
        """
        # ensure prefix sum is the correct type and is contiguous
        prefix_sum = np.ascontiguousarray(prefix_sum, dtype=self.dtype)
        prefix_sum_flat = prefix_sum.ravel()
        # init return array
        flat_idx = np.zeros(prefix_sum.size, dtype=np.intp)
        # get ids
        get_prefix_sum_idx(flat_idx, prefix_sum_flat, self._flat_base, self._sumtree)
        # output shape should be same as prefix_sum shape
        if prefix_sum.ndim > 1:
            output = flat_idx.reshape(prefix_sum.shape)
        else:
            output = flat_idx
        if self.ndim <= 1 or flatten_indices:
            return output
        else:
            return np.unravel_index(output, self.shape)

    def sample(self, nsamples=1, flatten_indices=False):
        """
        Return a sample of indices, where the probability an index being
        sampled is equal to the value at that index divided by the
        sum of the array, i.e. ``probs = self/self.sum()``.

        Parameters
        ----------
        nsamples : int
            Number of samples to return
        flatten_indices : bool, optional
            Defaults to ``False``.  When ``True``, returns the indices
            as if ``self`` is a 1d array (i.e. ``self.ravel()``)

        Returns
        -------
        sample_of_indices : ndarray, tuple(ndarray)
            A new array holding the result is returned containing the
            sampled indices.  If the underlying ``SumTreeArray`` array
            is n-dimensional, where n>1, and ``flatten_indices=False``,
            then a tuple of ``n`` arrays is returned where each element of
            the tuple corresponds to each dimension of the ``SumTreeArray``
            array.

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        Examples
        --------
        >>> sum_tree = SumTreeArray(np.array([1,2,3,4],dtype='float32'))
        >>> sum_tree
        SumTreeArray([1., 2., 3., 4.], dtype=float32)
        >>> sum_tree.sample(10)
        array([2, 3, 3, 3, 3, 1, 2, 2, 2, 0], dtype=int32)
        >>> # probability of being sampled
        >>> sum_tree / sum_tree.sum()
        array([0.1, 0.2, 0.3, 0.4], dtype=float32)
        >>> # sampled proportions
        >>> (sum_tree.sample(1000)[None] == np.arange(4)[:,None]).mean(axis=1)
        array([0.10057, 0.19919, 0.29983, 0.40041])
        >>> # sampling from a 2-d array
        >>> sum_tree_from_2d_array = SumTreeArray(np.array([[1,2,3],[4,5,6]],dtype='int32'))
        >>> sum_tree_from_2d_array
        SumTreeArray([[1, 2, 3],
                      [4, 5, 6]], dtype=int32)
        >>> sum_tree_from_2d_array.sample(4)
        (array([1, 1, 1, 0]), array([1, 1, 2, 2]))
        >>> sum_tree_from_2d_array.sample(4,flatten_indices=True)
        array([4, 4, 5, 2], dtype=int32)
        """
        if self.sum() == 0:
            raise ValueError("array must have at least 1 positive value")
        # sample priority values in the cumulative sum
        vals = (self.sum() * np.random.rand(nsamples)).astype(self.dtype)
        return self.get_prefix_sum_id(vals, flatten_indices)

    def _parse_axis_arg(self, axis):
        if axis is None:
            return None
        else:
            axes = np.array(axis).reshape(-1)
            if len(set(axes)) < len(axes):
                raise IndexError(
                    "invalid axis argument: %s contains duplicates" % str(axis)
                )
            return np.arange(self.ndim)[axes]

    def sum(self, axis=None, keepdims=False):
        """
        Functions the same as ``numpy.sum(...)`` except that some ``sum``
        operations can be significantly faster for large arrays
        when the sum operation is performed over C-contiguous
        ranges of indices.  For example, for a 2d array, supplying
        ``axis=1`` to the ``sum`` function would be a sum operation over
        C-contiguous ranges of indices, and thus benefit from sum-tree
        speedups.  However, ``axis=0`` is not C-contiguous sum operation,
        and would thus revert to the standard ``numpy.sum`` method.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which a sum is performed. The default,
            ``axis=None``, will sum all of the elements of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, a sum is performed on all of the
            axes specified in the tuple instead of a single axis or all the
            axes as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in
            the result as dimensions with size one. With this option, the
            result will broadcast correctly against the input array. Defaults
            to ``False``.

        Returns
        -------
        sum_along_axis : ndarray
            An array with the same shape as self, with the specified axis
            or axes removed, unless ``keepdims=True`` in which case the
            specified axis or axes are not removed, but will have size ``1``.
            If axis is None, a scalar is returned.

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is raised on overflow.
        SumTreeArray does not use the improved precision techniques that NumPy uses when
        summing along the fast axis (C-contiguous), instead relying on the standard floating
        point precision arithmetic provided by Python.

        References
        ----------
        [1] NumPy.  For more information on ``numpy.sum``, please see
        https://numpy.org/doc/stable/reference/generated/numpy.sum.html

        Examples
        --------
        >>> x = SumTreeArray(np.ones((1000,1000)))
        >>> # lightning fast
        >>> x.sum()
        1000000.0
        >>> # very fast
        >>> x.sum(axis=1)
        array([1000., ..., 1000.], dtype='float64')
        >>> # slightly slower than NumPy
        >>> x.sum(axis=0)
        array([1000., ..., 1000.], dtype='float64')
        >>> # maintain dims with keepdims=True
        >>> x.sum(axis=1,keepdims=True).shape
        (1000, 1)
        >>> x.sum(axis=1).shape
        (1000,)
        """
        axes = self._parse_axis_arg(axis)
        if axes is None:
            if keepdims:
                return self._sumtree[1:2].reshape([1] * self.ndim)
            else:
                return self._sumtree[1]
        else:
            if axes.min() == self.ndim - len(axes):
                # strides are contiguous along leaves of sumtree
                stride = int(np.prod(np.array(self.shape)[axes]))
                output = strided_sum(self._flat_base, self._sumtree, stride)
                if keepdims:
                    return output.reshape(list(output.shape) + [1] * len(axes))
                else:
                    return output
            else:
                return super(SumTreeArray, self).sum(axis=axis, keepdims=keepdims)
