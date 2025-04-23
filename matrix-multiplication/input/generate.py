import random
import sys

def generate_matrices(N):
    filename = f"input_{N}_{N}_{N}_{N}.in"
    with open(filename, "w") as file:
        # Write the first line with N repeated
        file.write(" ".join([str(N)] * 4) + "\n")
        
        # Generate and write the matrix
        for i in range(N):
            row = [f"{random.uniform(0, 1):.2f}" for _ in range(N)]
            file.write(" ".join(row) + "\n")
            
        # Generate and write the matrix
        for i in range(N):
            row = [f"{random.uniform(0, 1):.2f}" for _ in range(N)]
            file.write(" ".join(row) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate.py <matrix_size>")
        sys.exit(1)
    
    try:
        N = int(sys.argv[1])
    except ValueError:
        print("Error: Matrix size must be an integer.")
        sys.exit(1)
    
    generate_matrices(N)
    print(f"Matrices written to input_{N}_{N}_{N}_{N}.in")