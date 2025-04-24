### Setup Initialization

To initialize the setup, execute the following command:

```bash
make init
```

This command will create the `output` and `bin` directories and generate the necessary input files in the `input` directory.

### Executing the Program

After making modifications to the code, you can run the program using the following command:

```bash
make run_<code-name>
```

For example, if your file is named `MATMUL_cuda_1.cu`, replace `<code-name>` with `cuda_1` and execute:

```bash
make run_cuda_1
```

By default, the program processes matrices of size 100x100. To modify the matrix size, update the `job.sh` file accordingly.

### Running Benchmarks

To perform benchmarking, use the following command:

```bash
make run_benchmark TARGET=<code-name> N_BENCHMARKS=<value> N_ITERATIONS=<value>
```

- **`N_BENCHMARKS`**: Specifies the size of the matrices to be generated and multiplied. It must be an integer where `N_BENCHMARKS >= 1`.
- **`N_ITERATIONS`**: Indicates the number of times the code will execute for each benchmark. It must be an integer where `N_ITERATIONS >= 1`. The program will calculate and display the average execution time across all iterations.

For instance, to benchmark with `code-name` set to `cuda_1`, a matrix size of 20x20, and 10 iterations, execute:

```bash
make run_benchmark TARGET=cuda_1 N_BENCHMARKS=20 N_ITERATIONS=10
```

### Adding a Custom Implementation of `do_compute`

To integrate a custom implementation, follow these steps:

1. Create a new file using the naming convention `MATMUL_<implementation-name>(.c/.cu)`.
2. Ensure that you import `KMEANS.h` and implement the `do_compute` method. For reference, review the `MATMUL_cuda_1.cu` file.
3. Open the `Makefile` and locate the sections marked with comments starting with `NEWFILE`.
4. Add references to your implementation in all locations marked with `NEWFILE` comments.
5. Execute the new implementation using the instructions provided above.