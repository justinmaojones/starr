# Benchmarks

| class         | set 100 values   | priority-sample 100 values   | sum entire array   |   array size | dtype   |
|:--------------|:-----------------|:-----------------------------|:-------------------|-------------:|:--------|
| NDArray       | 2 μs             | 11125 μs                     | 272 μs             |      1000000 | float64 |
| PrefixSumTree | 12 μs            | 52 μs                        | 1 μs               |      1000000 | float64 |
| CPlusPlus     | 4 μs             | 26 μs                        | 0 μs               |      1000000 | float64 |
| PythonList    | 1728 μs          | 739 μs                       | 0 μs               |      1000000 | float64 |