#!/bin/sh
#SBATCH --time=00:15:00               # Set the maximum runtime for the job (15 minutes)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU resource
#SBATCH --constraint=TitanX           # Specify a hardware constraint (e.g., TitanX GPU)

./bin/MATMUL_seq ./input/input_1.in                          # Execute the program
