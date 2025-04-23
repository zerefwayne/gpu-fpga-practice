#!/bin/sh
#SBATCH --time=00:15:00               # Set the maximum runtime for the job (15 minutes)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU resource
#SBATCH --constraint=TitanX           # Specify a hardware constraint (e.g., TitanX GPU)

# Validate input arguments

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <binary_file_path> <problem_size (S/M/L)>"
    exit 1
fi

if [ ! -x "$1" ]; then
    echo "Error: The binary file '$1' does not exist or is not executable."
    exit 1
fi

if [ "$2" != "S" ] && [ "$2" != "M" ] && [ "$2" != "L" ]; then
    echo "Error: Problem size must be one of 'S', 'M', or 'L'."
    exit 1
fi


# Run binaries for different sizes

if [ "$2" = "S" ] || [ "$2" = "M" ] || [ "$2" = "L" ]; then
    for i in {10..100..10}; do
        "$1" ./input/input_${i}_${i}_${i}_${i}.in
    done
fi

if [ "$2" = "M" ] || [ "$2" = "L" ]; then
    for i in {200..1000..100}; do
        "$1" ./input/input_${i}_${i}_${i}_${i}.in
    done
fi

if [ "$2" = "L" ]; then
    for i in {2000..4000..1000}; do
        "$1" ./input/input_${i}_${i}_${i}_${i}.in
    done
fi
