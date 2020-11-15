# Benchmarks

| class        | set 100 values   | priority-sample 100 values   | sum entire array   |   array size | dtype   |
|:-------------|:-----------------|:-----------------------------|:-------------------|-------------:|:--------|
| NDArray      | 2 μs             | 9974 μs                      | 253 μs             |      1000000 | float64 |
| SumTreeArray | 13 μs            | 45 μs                        | 1 μs               |      1000000 | float64 |
| CPlusPlus    | 4 μs             | 28 μs                        | 0 μs               |      1000000 | float64 |
| PythonList   | 1854 μs          | 827 μs                       | 0 μs               |      1000000 | float64 |