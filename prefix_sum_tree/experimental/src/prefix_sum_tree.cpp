#include <math.h>
#include <stdexcept>
#include <stdio.h>
#include "prefix_sum_tree.h"

namespace prefix_sum_tree {

    void update_tree_c(
            int idx, 
            double val, 
            double* array, 
            const int n, 
            double* sumtree) {

        if(val < 0){
            throw std::invalid_argument("val must be non-negative");
        }

        // assumes array and sumtree are the same size, where sumtree is a prefix sum tree of array
        array += idx;
        idx = (idx + n)/2; // idx of parent in sumtree
        double diff = val - *array;
        *array = val;
        sumtree += idx;
        while(idx > 0) {
            *sumtree += diff;
            sumtree -= idx - (idx/2); // move to parent (idx rounds down)
            idx /= 2; // idx rounds down
        }
    }

    void update_tree_multi_c(
            int* idxs, const int I,
            double* vals, const int V, 
            double* array, const int n,
            double* sumtree) {

        const double* vals0 = vals;
        int v = 0;
        for(int i=0; i<I; i++) {
            update_tree_c(*idxs,*vals,array,n,sumtree);
            idxs++;
            vals++;
            v++;

            // if V < I, cycle through vals
            if(v==V){ 
                vals = (double*) vals0;
                v = 0;
            }
        }
    }

    int get_prefix_sum_idx_c(
            double val, 
            double* array, 
            double* sumtree, 
            const int n) {

        int i = 1;
        double left_val;
        while(i < n) {
            int left = 2*i;
            if(left < n) {
                left_val = *(sumtree+left);
            } else {
                left_val = *(array+left-n);
            }
            if(val >= left_val) {
                i = left + 1; //right
                val -= left_val;
            } else {
                i = left;
            }
        }
        return i - n;
    }

    void get_prefix_sum_idx_multi_c(int* outarray, double* vals, const int m, double* array, const int n, double* sumtree) {
        for(int i=0; i<m; i++) {
            *outarray = get_prefix_sum_idx_c(*vals,array,sumtree,n);
            outarray++;
            vals++;
        }
    }
};



