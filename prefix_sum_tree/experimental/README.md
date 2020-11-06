# Benchmarks

| class         | set 100 values   | priority-sample 100 values   | sum entire array   |   array size | dtype   |
|:--------------|:-----------------|:-----------------------------|:-------------------|-------------:|:--------|
| NDArray       | 2 μs             | 9489 μs                      | 235 μs             |      1000000 | float64 |
| PrefixSumTree | 12 μs            | 56 μs                        | 1 μs               |      1000000 | float64 |
| CPlusPlus     | 4 μs             | 23 μs                        | 0 μs               |      1000000 | float64 |
| PythonList    | 1777 μs          | 802 μs                       | 0 μs               |      1000000 | float64 |