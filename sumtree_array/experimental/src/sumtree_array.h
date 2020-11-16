#ifndef PREFIX_SUM_TREE_H
#define PREFIX_SUM_TREE_H

namespace sumtree_array {

    void update_tree_multi_c(
            int* idxs, const int I,
            double* vals, const int V, 
            double* array, const int n,
            double* sumtree);

    void get_prefix_sum_idx_multi_c(
            int* outarray, 
            double* vals, 
            const int m, 
            double* array, 
            const int n, 
            double* sumtree);

}

#endif
