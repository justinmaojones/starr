import cython
import numpy as np

ctypedef fused INDEX_TYPE:
    cython.integral
    unsigned short
    unsigned int
    unsigned long

ctypedef fused ARRAY_TYPE:
    cython.integral
    cython.floating
    long double
    unsigned char
    unsigned short
    unsigned int
    unsigned long

cdef bint boolean_variable = True
    
cdef ARRAY_TYPE disjoint_get(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, INDEX_TYPE i):
    cdef INDEX_TYPE n = len(array)
    if i >= n:
        return array[i-n]
    else:
        return sumtree[i]

def update_prefix_sum_tree(
            INDEX_TYPE[:] idxs,
            ARRAY_TYPE[:] vals,
            ARRAY_TYPE[:] array,
            ARRAY_TYPE[:] sumtree):

    if len(array) != len(sumtree):
        raise TypeError, "array and sumtree must have the same size"

    cdef int i # index of idxs and vals
    cdef INDEX_TYPE idx # index of array and sumtree
    cdef INDEX_TYPE n = <INDEX_TYPE>array.shape[0] # size of array and sumtree
    cdef INDEX_TYPE nv = <INDEX_TYPE>vals.shape[0] # size of vals 
    cdef ARRAY_TYPE diff 

    for i in range(idxs.shape[0]):
        # assumes array and sumtree are the same size, where sumtree is a prefix sum tree of array
        idx = idxs[i]
        diff = vals[i % nv] - array[idx]
        array[idx] += diff
        idx = (idx + n) / 2 # index of parent in sumtree
        while idx > 0:
            sumtree[idx] += diff
            idx -= idx - (idx / 2) # move to parent 

def build_sumtree_from_array(
            ARRAY_TYPE[:] array,
            ARRAY_TYPE[:] sumtree):

    if len(array) != len(sumtree):
        raise TypeError, "array and sumtree must have the same size"

    cdef int i 
    for i in range(len(array)-1,0,-1):
        sumtree[i] = disjoint_get(array, sumtree, i*2) + disjoint_get(array, sumtree, i*2+1)

def get_prefix_sum_idx(
            INDEX_TYPE[:] output,
            ARRAY_TYPE[:] vals,
            ARRAY_TYPE[:] array,
            ARRAY_TYPE[:] sumtree):

    if len(output) != len(vals):
        raise TypeError, "output and vals must have the same size"
    if len(array) != len(sumtree):
        raise TypeError, "array and sumtree must have the same size"

    cdef INDEX_TYPE N = <INDEX_TYPE>array.shape[0] # size of array and sumtree
    cdef INDEX_TYPE M = <INDEX_TYPE>vals.shape[0] # size of output and vals 
    cdef INDEX_TYPE i # current search index 
    cdef INDEX_TYPE left # index of left subtree
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

cdef INDEX_TYPE power_of_2(INDEX_TYPE i):
    cdef INDEX_TYPE y = 0
    while i > 0:
        i >>= 1
        y += 1
    return y

cdef ARRAY_TYPE sum_in_c(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, INDEX_TYPE a, INDEX_TYPE b):

    b -= 1 # not-inclusive of right end-point
    if a == b:
        return array[a]
    if a > b:
        return 0
    
    cdef INDEX_TYPE n = len(array)
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
            

def sum(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, INDEX_TYPE index_from, INDEX_TYPE index_to):
    return sum_in_c(array, sumtree, index_from, index_to)

cdef void strided_sum_in_c(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, ARRAY_TYPE[:] output, INDEX_TYPE stride):

    if len(array) // stride not in (len(output), len(output)-1):
        raise TypeError, "invalid output size"

    cdef INDEX_TYPE i 
    for i in range(len(output)):
        output[i] = sum_in_c(array, sumtree, stride*i, min(stride*(i+1),len(array)))

def strided_sum(ARRAY_TYPE[:] array, ARRAY_TYPE[:] sumtree, INDEX_TYPE stride):
    
    # allows for strides that don't divide evenly into length of array
    n = len(array) // stride
    if len(array) % stride != 0:
        n += 1

    cdef ARRAY_TYPE[:] output = np.empty_like(array[:n])
    strided_sum_in_c(array, sumtree, output, stride)
    return np.array(output)

