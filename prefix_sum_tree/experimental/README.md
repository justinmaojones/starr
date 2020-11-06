# Benchmarks

| class         | set 100 values   | priority-sample 100 values   | sum entire array   |   array size | dtype   |
|:--------------|:-----------------|:-----------------------------|:-------------------|-------------:|:--------|
| NDArray       | 2 μs             | 9435 μs                      | 253 μs             |      1000000 | float64 |
| PrefixSumTree | 12 μs            | 51 μs                        | 1 μs               |      1000000 | float64 |
| CPlusPlus     | 4 μs             | 22 μs                        | 0 μs               |      1000000 | float64 |
| PythonList    | 1712 μs          | 765 μs                       | 0 μs               |      1000000 | float64 |