#!/bin/sh
#SBATCH --time=00:15:00               # Set the maximum runtime for the job (15 minutes)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU resource
#SBATCH --constraint=TitanX           # Specify a hardware constraint (e.g., TitanX GPU)

# Validate input arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <BENCHMARK_BINARY> <TARGET_BINARY> <N_BENCHMARKS> <N_ITERATIONS>"
    echo "Example: $0 ./bin/MATMUL_benchmark ./bin/MATMUL_cuda_1.o 20 10"
    exit 1
fi

BENCHMARK_BINARY="$1"
TARGET_BINARY="$2"
N_BENCHMARKS="$3"
N_ITERATIONS="$4"

# Ensure BENCHMARK_BINARY exists and is executable
if [ ! -x "$BENCHMARK_BINARY" ]; then
    echo "Error: BENCHMARK_BINARY '$BENCHMARK_BINARY' does not exist or is not executable."
    exit 1
fi

# Ensure N_BENCHMARKS is a positive integer
if ! [ "$N_BENCHMARKS" -gt 1 ] 2>/dev/null; then
    echo "Error: N_BENCHMARKS '$N_BENCHMARKS' must be a positive integer."
    exit 1
fi

# Ensure N_ITERATIONS is a positive integer
if ! [ "$N_ITERATIONS" -gt 1 ] 2>/dev/null; then
    echo "Error: N_ITERATIONS '$N_ITERATIONS' must be a positive integer."
    exit 1
fi

echo "TARGET: $2"
$BENCHMARK_BINARY $N_BENCHMARKS $N_ITERATIONS
