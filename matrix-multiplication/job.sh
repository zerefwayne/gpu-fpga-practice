#!/bin/sh
#SBATCH --time=00:15:00               # Set the maximum runtime for the job (15 minutes)
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --gres=gpu:1                  # Request 1 GPU resource
#SBATCH --constraint=TitanX           # Specify a hardware constraint (e.g., TitanX GPU)


# ./bin/MATMUL_seq ./input/input_2_2_2_2.in

for i in {10..100..10}; do
    ./bin/MATMUL_seq ./input/input_${i}_${i}_${i}_${i}.in
done

for i in {200..1000..100}; do
    ./bin/MATMUL_seq ./input/input_${i}_${i}_${i}_${i}.in
done

# for i in {2000..4000..1000}; do
#     ./bin/MATMUL_seq ./input/input_${i}_${i}_${i}_${i}.in
# done
