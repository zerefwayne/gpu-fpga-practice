#!/bin/sh
#SBATCH --time=00:15:00               # Set the maximum runtime for the job (15 minutes)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU resource
#SBATCH --constraint=TitanX           # Specify a hardware constraint (e.g., TitanX GPU)

# Validate input arguments
if [ -z "$1" ]; then
    echo "Error: No executable provided."
    echo "Usage: $0 <executable>"
    exit 1
fi

SIZE=100

"$1" ./input/input_${SIZE}_${SIZE}_${SIZE}_${SIZE}.in
