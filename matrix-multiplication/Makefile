# Define compilers
CUDACC = nvcc

# Define flags
CFLAGS = -O3 -Wall
CUDAFLAGS = -O3
LIBS = -lm -lcublas

# Define directories
BIN_DIR = bin
OUTPUT_DIR = output

# Define object files
MATMUL_OBJ            = $(BIN_DIR)/MATMUL.o
MATMUL_VALIDATE_OBJ   = $(BIN_DIR)/MATMUL_validate.o
MATMUL_SEQ_OBJ        = $(BIN_DIR)/MATMUL_seq.o

# Define final binary
TARGET = $(BIN_DIR)/MATMUL_seq

# Final list of object files
OBJS = $(MATMUL_VALIDATE_OBJ) $(MATMUL_SEQ_OBJ) $(MATMUL_OBJ)  

# Default target
all: $(TARGET) 

init_all: $(BIN_DIR) $(OUTPUT_DIR) GENERATE_INPUT

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

GENERATE_INPUT:
	cd input && sh ./generate.sh

# Compile MATMUL_validate.cu
$(MATMUL_VALIDATE_OBJ): MATMUL_validate.cu | $(BIN_DIR)
	$(CUDACC) $(CUDAFLAGS) -c MATMUL_validate.cu -o $(MATMUL_VALIDATE_OBJ)

# Compile MATMUL_seq.c with nvcc (as C code)
$(MATMUL_SEQ_OBJ): MATMUL_seq.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c MATMUL_seq.c -o $(MATMUL_SEQ_OBJ)

# Compile MATMUL.c with nvcc (as C code)
$(MATMUL_OBJ): MATMUL.c | $(BIN_DIR)
	$(CUDACC) -x cu $(CUDAFLAGS) -c MATMUL.c -o $(MATMUL_OBJ)

# Link all objects into the final binary
$(TARGET): $(OBJS)
	$(CUDACC) $(OBJS) $(LIBS) -o $(TARGET)

# Clean build files
clean:
	rm -rf $(BIN_DIR)/* ./slurm*.out

# Full clean, build, and run
init:
	make --no-print-directory clean && make --no-print-directory init_all

run:
	make --no-print-directory clean && make --no-print-directory all && sbatch job.sh
