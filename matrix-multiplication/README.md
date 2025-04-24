# Matrix Multiplication Benchmark Suite

This project provides a framework for benchmarking matrix multiplication implementations on various platforms such as CPUs and GPUs. The suite supports custom implementations and allows users to measure performance across different configurations.

---

## Setup

To initialize the project environment, execute the following command:

```bash
make init
```

This will create the necessary directories (`output` and `bin`) and generate sample input files in the `input` directory.

---

## Executing the Program

After modifying or adding code, you can execute the program using the following command:

```bash
make run TARGET=<implementation-name>
```

### Example:
If your implementation file is named `MATMUL_cuda_1.cu`, replace `<implementation-name>` with `cuda_1`:

```bash
make run TARGET=cuda_1
```

By default, the program processes matrices of size **100x100**. To modify the matrix size, update the `job.sh` file accordingly.

---

## Running Benchmarks

To benchmark a specific implementation, use the following command:

```bash
make benchmark TARGET=<implementation-name> N_BENCHMARKS=<value> N_ITERATIONS=<value>
```

### Parameters:
- **`TARGET`**: The name of the implementation to benchmark (e.g., `cuda_1` for `MATMUL_cuda_1.cu`).
- **`N_BENCHMARKS`**: Specifies the number of matrix sizes to benchmark. Must be an integer where `N_BENCHMARKS >= 1`.
- **`N_ITERATIONS`**: Specifies the number of iterations for each benchmark. Must be an integer where `N_ITERATIONS >= 1`. The program computes and displays the average execution time across all iterations.

### Example:
To benchmark the `cuda_1` implementation with 20 matrix sizes and 10 iterations, run:

```bash
make benchmark TARGET=cuda_1 N_BENCHMARKS=20 N_ITERATIONS=10
```

---

## Adding a Custom Implementation

To add a custom implementation of the `do_compute` function, follow these steps:

1. Create a new file using the naming convention `MATMUL_<implementation-name>.c` or `MATMUL_<implementation-name>.cu`.
2. Include the `MATMUL.h` header file in your implementation.
3. Implement the `do_compute` function. Refer to existing implementations (e.g., `MATMUL_cuda_1.cu`) for guidance.
4. Use the `make run` or `make benchmark` commands to execute or benchmark your implementation.

---

## Notes

- Ensure that all dependencies are correctly installed and accessible.
- Use the `make clean_all` command to remove all input files, binaries and outputs. Check `Makefile` for more options.
- For advanced configurations, modify the `Makefile`, `run_job.sh` or `benchmark_job.sh` script as required.
