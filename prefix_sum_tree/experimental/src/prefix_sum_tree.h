#ifndef PREFIX_SUM_TREE_H
#define PREFIX_SUM_TREE_H

namespace prefix_sum_tree {
    void update_tree_c(int idx, double val, double* array, const int n);
    void update_tree_multi_c(int* idxs, double* vals, const int m, double* array, const int n);
    void update_disjoint_tree_multi_c(
            int* idxs, const int I,
            double* vals, const int V, 
            double* array, const int n,
            double* sumtree);
    int get_prefix_sum_idx_c(double val, double* array, const int n);
    void get_prefix_sum_idx_multi_c(int* outarray, double* vals, const int m, double* array, const int n); 
    void get_prefix_sum_idx_disjoint_multi_c(int* outarray, double* vals, const int m, double* array, const int n, double* sumtree);
}

#endif
