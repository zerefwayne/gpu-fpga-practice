# Define compilers
CC=gcc
CUDACC=nvcc

# Define flags
CFLAGS=-O3 -Wall
CUDAFLAGS=-O3
LIBS=-lm

# Define output directory
BIN_DIR=bin
OUTPUT_DIR=output

# Define objects
OBJS=$(BIN_DIR)/MATMUL

all: $(BIN_DIR) $(OUTPUT_DIR) $(OBJS)

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Create output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Define objects for base class
$(BIN_DIR)/MATMUL.o: MATMUL.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(DEBUG) -c $< -o $@

$(BIN_DIR)/MATMUL: $(BIN_DIR)/MATMUL.o
	$(CC) $(DEBUG) $< $(LIBS) -o $@

clean:
	rm -rf $(BIN_DIR)/* slurm*.out
	rm -rf $(OUTPUT_DIR)/*

run:
	make --no-print-directory clean && make --no-print-directory all && sbatch job.sh
