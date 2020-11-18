# Benchmarks

| class        | set 100 values   | priority-sample 100 values   | sum entire array   |   array size | dtype   |
|:-------------|:-----------------|:-----------------------------|:-------------------|-------------:|:--------|
| NDArray      | 2 μs             | 14828 μs                     | 364 μs             |      1000000 | float64 |
| SumTreeArray | 13 μs            | 57 μs                        | 1 μs               |      1000000 | float64 |
| CPlusPlus    | 4 μs             | 27 μs                        | 0 μs               |      1000000 | float64 |
| PythonList   | 1987 μs          | 840 μs                       | 0 μs               |      1000000 | float64 |