import prefix_sum_tree
import prefix_sum_tree.experimental
import numpy as np
import timeit

K = 1000
N = 100000000
S = 1000
np.random.seed(1)
IDX = np.random.choice(N,size=K).astype(np.int32)
VALS = np.random.choice(10,size=K).astype(float)

class TimingTest(object):

    def __init__(self):
        self.base = np.zeros(N)
        self.sumtree = np.zeros(N)

    def test_set(self):
        raise NotImplementError

    def test_sample(self):
        raise NotImplementError

    def test_sum(self):
        raise NotImplementError

class NDArray(TimingTest):

    def __init__(self):
        self.base = np.zeros(N)

    def test_set(self):
        self.base[IDX] = VALS

    def test_sum(self):
        self.base.sum()


class TimePrefixSumTree(TimingTest):

    def __init__(self):
        self.sumtree = prefix_sum_tree.PrefixSumTree(np.zeros(N))
        self.test_set() # initialize with vals

    def test_set(self):
        self.sumtree[IDX] = VALS

    def test_sample(self, nsamples=S):
        return self.sumtree.sample(nsamples)

    def test_sum(self, nsamples=S):
        return self.sumtree.sum()

class TimeSlowExperimental(TimingTest):

    def __init__(self):
        self.sumtree = prefix_sum_tree.experimental.slow.SumTree(N)
        self.test_set() # initialize with vals

    def test_set(self):
        for i, v in zip(IDX, VALS):
            self.sumtree[i] = v

    def test_sample(self, nsamples=S):
        return self.sumtree.sample(nsamples)

class TimeFastExperimental(TimingTest):

    def __init__(self):
        self.base = np.zeros(N)
        self.sumtree = np.zeros(N)
        self.test_set() # initialize with vals

    def test_set(self):
        prefix_sum_tree.experimental.update_disjoint_tree_multi(
            IDX, VALS, self.base, self.sumtree)

    def test_sample(self, nsamples=S):
        vals_search = (self.sumtree[1] * np.random.rand(nsamples)).astype(self.sumtree.dtype)
        output = np.zeros(nsamples,dtype=np.int32)
        return prefix_sum_tree.experimental.get_prefix_sum_multi_idx2(
            output, vals_search, self.base, self.sumtree)

def test_set(class_name, num_execs = 1000):
    setup = "from __main__ import {class_name}; timing_test = {class_name}()".format(**locals())
    total_duration = timeit.timeit("timing_test.test_set()", setup=setup, number=num_execs)
    duration_str = u"%0.1f \u03BCs" % (total_duration/num_execs*1e6)
    print("{class_name}: {duration_str}".format(**locals()))

def test_sample(class_name, num_execs = 1000):
    setup = "from __main__ import {class_name}; timing_test = {class_name}()".format(**locals())
    total_duration = timeit.timeit("timing_test.test_sample()", setup=setup, number=num_execs)
    duration_str = u"%0.0f \u03BCs" % (total_duration/num_execs*1e6)
    print("{class_name}: {duration_str}".format(**locals()))

def test_sum(class_name, num_execs = 1000):
    setup = "from __main__ import {class_name}; timing_test = {class_name}()".format(**locals())
    total_duration = timeit.timeit("timing_test.test_sum()", setup=setup, number=num_execs)
    duration_str = u"%0.0f \u03BCs" % (total_duration/num_execs*1e6)
    print("{class_name}: {duration_str}".format(**locals()))


if __name__ == '__main__':

    print("\n>> __set__ %d values:\n" % N)
    test_set("NDArray", 1000)
    test_set("TimePrefixSumTree", 1000)
    test_set("TimeFastExperimental", 1000)
    test_set("TimeSlowExperimental", 100)

    print("\n>> priority-sample %d values:\n" % S)
    test_sample("TimePrefixSumTree", 1000)
    test_sample("TimeFastExperimental", 1000)
    test_sample("TimeSlowExperimental", 100)

    print("\n>> sum entire array:\n")
    test_sum("NDArray", 1000)
    test_sum("TimePrefixSumTree", 1000)

    print('')

