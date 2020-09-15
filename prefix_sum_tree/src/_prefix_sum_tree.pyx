import cython

ctypedef fused INDEX_TYPE:
    cython.integral

ctypedef fused ARRAY_TYPE:
    cython.integral
    cython.floating
    long double

def update_prefix_sum_tree(
            INDEX_TYPE[:] idxs,
            ARRAY_TYPE[:] vals,
            ARRAY_TYPE[:] array,
            ARRAY_TYPE[:] sumtree):

    assert idxs.shape[0] == vals.shape[0]
    assert array.shape[0] == sumtree.shape[0]

    cdef int i # index of idxs and vals
    cdef INDEX_TYPE idx # index of array and sumtree
    cdef INDEX_TYPE n = <INDEX_TYPE>array.shape[0] # size of array and sumtree
    cdef ARRAY_TYPE diff 

    for i in range(idxs.shape[0]):
        # assumes array and sumtree are the same size, where sumtree is a prefix sum tree of array
        idx = idxs[i]
        diff = vals[i] - array[idx]
        array[idx] += diff
        idx = (idx + n) / 2 # index of parent in sumtree
        while idx > 0:
            sumtree[idx] += diff
            idx -= idx - (idx / 2) # move to parent 


def get_prefix_sum_idx(
            INDEX_TYPE[:] output,
            ARRAY_TYPE[:] vals,
            ARRAY_TYPE[:] array,
            ARRAY_TYPE[:] sumtree):

    assert output.shape[0] == vals.shape[0]
    assert array.shape[0] == sumtree.shape[0]

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


