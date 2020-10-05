# global variable declaration allows package to be initialized through cython
__DUMMY__ = True

# cimport the Cython declarations for numpy
cimport numpy as cnp
import numpy as np

# if you want to use the Numpy-C-API from Cython
cnp.import_array()

cdef extern from "prefix_sum_tree.h" namespace "prefix_sum_tree" nogil:
    void update_tree_multi_c(
            int* idxs, const int I,
            double* vals, const int V, 
            double* array, const int n,
            double* sumtree);

cdef extern from "prefix_sum_tree.h" namespace "prefix_sum_tree" nogil:
    void get_prefix_sum_idx_multi_c(int* outarray, double* vals, const int m, double* array, const int n, double* sumtree);


def update_tree_multi(
            cnp.ndarray[int, ndim=1, mode="c"] idxs not None,
            cnp.ndarray[double, ndim=1, mode="c"] vals not None,
            cnp.ndarray[double, ndim=1, mode="c"] array not None,
            cnp.ndarray[double, ndim=1, mode="c"] sumtree not None):

    cdef int n = array.shape[0];
    cdef int I = idxs.shape[0];
    cdef int V = vals.shape[0];

    update_tree_multi_c(&idxs[0], I, &vals[0], V, &array[0], n, &sumtree[0]);

def get_prefix_sum_idx_multi(
            cnp.ndarray[int, ndim=1, mode="c"] output not None,
            cnp.ndarray[double, ndim=1, mode="c"] vals not None,
            cnp.ndarray[double, ndim=1, mode="c"] array not None,
            cnp.ndarray[double, ndim=1, mode="c"] sumtree not None):

    cdef int n = sumtree.shape[0];
    cdef int m = vals.shape[0];

    get_prefix_sum_idx_multi_c(&output[0], &vals[0], m, &array[0], n, &sumtree[0]);

    return output

