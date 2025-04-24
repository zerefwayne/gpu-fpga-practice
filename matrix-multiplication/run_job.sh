#!/bin/sh
#SBATCH --time=00:15:00               # Set the maximum runtime for the job (15 minutes)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU resource
#SBATCH --constraint=TitanX           # Specify a hardware constraint (e.g., TitanX GPU)

# Validate input arguments
if [ "$#" -ne 2 ]; then
    echo "Error: Invalid number of arguments."
    echo "Usage: $0 <executable> <input_file>"
    exit 1
fi

TARGET_BINARY="$1"
INPUT_FILE="$2"

if [ ! -x "$TARGET_BINARY" ]; then
    echo "Error: Target executable '$TARGET_BINARY' is not found or not executable."
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

"$TARGET_BINARY" "$INPUT_FILE"
