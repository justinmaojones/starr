# Benchmarks

| class        | set 100 values   | sample 100 indices   | sum entire array   |   array size | dtype   |
|:-------------|:-----------------|:---------------------|:-------------------|-------------:|:--------|
| NDArray      | 2 μs             | 10431 μs             | 272 μs             |      1000000 | float64 |
| SumTreeArray | 13 μs            | 48 μs                | 1 μs               |      1000000 | float64 |
| PythonList   | 1755 μs          | 808 μs               | 0 μs               |      1000000 | float64 |