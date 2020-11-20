import starr
import starr.experimental
import pandas as pd
import numpy as np
import timeit

N = int(1e6)  # size of base array
K = int(1e2)  # number of indices to set
S = int(1e2)  # number of indices to sample
np.random.seed(1)
IDX = np.random.choice(N, size=K).astype(np.int32)
VALS = np.random.choice(10, size=K).astype(float)


class TimingTest(object):
    def __init__(self):
        self.base = np.ones(N)
        self.sumtree = np.ones(N)

    def test_set(self):
        raise NotImplementedError

    def test_sample(self, nsamples=S):
        raise NotImplementedError

    def test_sum(self):
        raise NotImplementedError


class NDArray(TimingTest):
    def __init__(self):
        self.base = np.ones(N)

    def test_set(self):
        self.base[IDX] = VALS

    def test_sample(self, nsamples=S):
        p = self.base / self.base.sum()
        return np.random.choice(len(self.base), size=S, p=p)

    def test_sum(self):
        self.base.sum()


class SumTreeArray(TimingTest):
    def __init__(self):
        self.sumtree = starr.SumTreeArray(np.ones(N))
        self.test_set()  # initialize with vals

    def test_set(self):
        self.sumtree[IDX] = VALS

    def test_sample(self, nsamples=S):
        return self.sumtree.sample(nsamples)

    def test_sum(self, nsamples=S):
        return self.sumtree.sum()


class PythonList(TimingTest):
    def __init__(self):
        self.sumtree = starr.experimental.slow.SumTree(N)
        self.test_set()  # initialize with vals

    def test_set(self):
        for i, v in zip(IDX, VALS):
            self.sumtree[i] = v

    def test_sample(self, nsamples=S):
        return self.sumtree.sample(nsamples)

    def test_sum(self):
        return self.sumtree.sum()


class CPlusPlus(TimingTest):
    def __init__(self):
        self.base = np.ones(N)
        self.sumtree = np.ones(N)
        self.test_set()  # initialize with vals

    def test_set(self):
        starr.experimental.update_tree_multi(IDX, VALS, self.base, self.sumtree)

    def test_sample(self, nsamples=S):
        vals_search = (self.sumtree[1] * np.random.rand(nsamples)).astype(
            self.sumtree.dtype
        )
        output = np.ones(nsamples, dtype=np.int32)
        return starr.experimental.get_prefix_sum_idx_multi(
            output, vals_search, self.base, self.sumtree
        )

    def test_sum(self):
        return self.sumtree[1]


def benchmark(class_name, func_name, num_execs=100):
    setup = "from __main__ import {class_name}; timing_test = {class_name}()".format(
        **locals()
    )
    total_duration = timeit.timeit(
        "timing_test.{func_name}()".format(**locals()), setup=setup, number=num_execs
    )
    duration_str = u"%0.0f \u03BCs" % (total_duration / num_execs * 1e6)
    return duration_str


if __name__ == "__main__":

    test_funcs = {
        "set %d values" % K: "test_set",
        "sample %d indices" % S: "test_sample",
        "sum entire array": "test_sum",
    }

    class_names = [
        "NDArray",
        "SumTreeArray",
        # "CPlusPlus",
        "PythonList",
    ]

    all_output = []
    for class_name in class_names:
        output = {"class": class_name}
        for fname in test_funcs:
            output[fname] = benchmark(class_name, test_funcs[fname])
        all_output.append(output)

    df = pd.DataFrame(all_output).set_index("class", drop=True)
    df["array size"] = N
    df["dtype"] = VALS.dtype

    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "README.md"), "w") as f:
        f.write("# Benchmarks\n\n")
        f.write(df.to_markdown())
        print("\n" + df.to_markdown() + "\n")
