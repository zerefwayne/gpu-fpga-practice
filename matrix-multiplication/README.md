### Setup Initialization

To initialize the setup, run the following command:

```bash
make init
```

This command will create the `output` and `bin` directories and generate input files in the `input` directory.

### Running After Making Changes

To execute the program after making changes, use the following command:

```bash
make run_<code-name>
```

For example, if your file is named `MATMUL_cuda_1.cu`, set `code-name` to `cuda_1` and run:

```bash
make run_cuda_1
```

By default, the program operates on matrices of size 100x100. To modify the matrix size, update the `job.sh` file accordingly.

### Running Benchmarks

To run benchmarks, use the following command:

```bash
make run_benchmark TARGET=<code-name> N_BENCHMARKS=<value> N_ITERATIONS=<value>
```

- **`N_BENCHMARKS`**: Specifies the size of the matrices to be created and multiplied. It must be an integer where `N_BENCHMARKS >= 1`.
- **`N_ITERATIONS`**: Specifies the number of times the code will execute for each benchmark. It must be an integer where `N_ITERATIONS >= 1`. The program will output the average execution time across all iterations.

For example, to run a benchmark with `code-name` set to `cuda_1`, matrix size of 20x20, and 10 iterations, use:

```bash
make run_benchmark TARGET=cuda_1 N_BENCHMARKS=20 N_ITERATIONS=10
```
