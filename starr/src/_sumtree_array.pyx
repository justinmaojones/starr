import cython
import numpy as np

ctypedef fused ARRAY_TYPE:
    cython.integral
    cython.floating
    long double
    unsigned char
    unsigned short
    unsigned int
    unsigned long

cdef bint boolean_variable = True
    
cdef ARRAY_TYPE disjoint_get(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, Py_ssize_t i):
    cdef Py_ssize_t n = len(array)
    if i >= n:
        return array[i-n]
    else:
        return sumtree[i]

def update_sumtree(
            Py_ssize_t[:] idxs,
            ARRAY_TYPE[:] vals,
            ARRAY_TYPE[:] array,
            ARRAY_TYPE[:] sumtree):

    if len(array) != len(sumtree):
        raise TypeError, "array and sumtree must have the same size"

    cdef int i # index of idxs and vals
    cdef Py_ssize_t idx # index of array and sumtree
    cdef Py_ssize_t n = <Py_ssize_t>array.shape[0] # size of array and sumtree
    cdef Py_ssize_t nv = <Py_ssize_t>vals.shape[0] # size of vals 
    cdef ARRAY_TYPE diff 

    for i in range(len(vals)):
        if vals[i] < 0:
            raise ValueError, "vals must be non-negative"

    for i in range(idxs.shape[0]):
        # assumes array and sumtree are the same size, where sumtree is a prefix sum tree of array
        idx = idxs[i]
        diff = vals[i % nv] - array[idx]
        array[idx] += diff
        idx = (idx + n) // 2 # index of parent in sumtree
        while idx > 0:
            sumtree[idx] += diff
            idx -= idx - (idx // 2) # move to parent 

def build_sumtree_from_array(
            ARRAY_TYPE[:] array,
            ARRAY_TYPE[:] sumtree):

    if len(array) != len(sumtree):
        raise TypeError, "array and sumtree must have the same size"

    if min(array) < 0:
        raise ValueError, "array elements must be non-negative"

    # iterate upwards through tree, stopping at root node at sumtree[1]
    cdef int i 
    for i in range(len(array)-1,0,-1):
        sumtree[i] = disjoint_get(array, sumtree, i*2) + disjoint_get(array, sumtree, i*2+1)

def get_prefix_sum_idx(
            Py_ssize_t[:] output,
            ARRAY_TYPE[:] vals,
            ARRAY_TYPE[:] array,
            ARRAY_TYPE[:] sumtree):

    if len(output) != len(vals):
        raise TypeError, "output and vals must have the same size"
    if len(array) != len(sumtree):
        raise TypeError, "array and sumtree must have the same size"

    cdef Py_ssize_t N = <Py_ssize_t>array.shape[0] # size of array and sumtree
    cdef Py_ssize_t M = <Py_ssize_t>vals.shape[0] # size of output and vals 
    cdef Py_ssize_t i # current search index 
    cdef Py_ssize_t left # index of left subtree
    cdef ARRAY_TYPE prefix_sum # prefix_sum to search for
    cdef ARRAY_TYPE left_val # prefix_sum of left subtree

    for m in range(M):

        # search for index of array such that sum(array[:index]) < prefix_sum
        prefix_sum = vals[m]

        # starting index in the sumtree
        i = 1; 

        # once i >= N, it has found a leaf node in array
        while(i < N): 

            # index of left subtree
            left = 2*i;

            # if not a leaf, get sum of left subtree in sumtree
            if left < N:
                left_val = sumtree[left]

            # get value of leaf in array
            else:
                left_val = array[left-N];

            # if prefix sum >= sum of left subtree, then search right subtree
            if prefix_sum >= left_val:

                # right subtree
                i = left + 1 

                # shift prefix_sum to right subtree
                prefix_sum -= left_val

            # search left subtree
            else:
                i = left

        # return corresponding leaf node in array
        output[m] = i - N

    return output

cdef Py_ssize_t power_of_2(Py_ssize_t i):
    cdef Py_ssize_t y = 0
    while i > 0:
        i >>= 1
        y += 1
    return y

cdef ARRAY_TYPE sum_over_in_c(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, Py_ssize_t a, Py_ssize_t b):

    b -= 1 # not-inclusive of right end-point
    if a == b:
        return array[a]
    if a > b:
        return 0
    
    cdef Py_ssize_t n = len(array)
    a += n
    b += n
    cdef ARRAY_TYPE subtract = 0

    # false if a and b share the same largest bit, otherwise true
    # indicates whather a and b are at the same depth of the sumtree
    cdef bint wrapped = a < a ^ b     

    if wrapped:
        # b is one depth lower in tree, so do one
        # step for just b, which brings both a and b
        # to the same level in the tree
        if b % 2 == 0:
            # is left node, subtract right
            subtract += disjoint_get(array, sumtree, b+1)
        b //= 2
        
    while (a // 2) != (b // 2):
        if a % 2 == 1:
            # is right node, subtract left
            subtract += disjoint_get(array, sumtree, a-1)
        if b % 2 == 0:
            # is left node, subtract right
            subtract += disjoint_get(array, sumtree, b+1)
        a //= 2
        b //= 2
        
    if wrapped:
        return sumtree[1] - subtract
    else:
        return disjoint_get(array, sumtree, a // 2) - subtract
            

def sum_over(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, Py_ssize_t index_from, Py_ssize_t index_to):
    return sum_over_in_c(array, sumtree, index_from, index_to)

cdef void strided_sum_in_c(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, ARRAY_TYPE[:] output, Py_ssize_t stride):

    if len(array) // stride not in (len(output), len(output)-1):
        raise TypeError, "invalid output size"

    cdef Py_ssize_t i 
    for i in range(len(output)):
        output[i] = sum_over_in_c(array, sumtree, stride*i, min(stride*(i+1),len(array)))

def strided_sum(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, Py_ssize_t stride):
    
    # allows for strides that don't divide evenly into length of array
    n = len(array) // stride
    if len(array) % stride != 0:
        n += 1

    cdef ARRAY_TYPE[:] output = np.empty_like(array[:n])
    strided_sum_in_c(array, sumtree, output, stride)
    return np.array(output)

