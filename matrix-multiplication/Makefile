# Define compilers
CUDACC = nvcc

# Define flags
CFLAGS = -O3 -Wall
CUDAFLAGS = -O3
LIBS = -lm -lcublas

# Define directories
BIN_DIR = bin
OUTPUT_DIR = output
INPUT_DIR = input

# Define object files
MATMUL_OBJ            = $(BIN_DIR)/MATMUL.o
MATMUL_VALIDATE_OBJ   = $(BIN_DIR)/MATMUL_validate.o
MATMUL_SEQ_OBJ        = $(BIN_DIR)/MATMUL_seq.o
MATMUL_CUDA_1_OBJ 	  = $(BIN_DIR)/MATMUL_cuda_1.o

# Define final binaries
TARGET_MATMUL_SEQ = $(BIN_DIR)/MATMUL_seq
TARGET_MATMUL_CUDA_1 = $(BIN_DIR)/MATMUL_cuda_1

# Final list of object files
OBJS_SEQ = $(MATMUL_VALIDATE_OBJ) $(MATMUL_SEQ_OBJ) $(MATMUL_OBJ)  
OBJS_CUDA_1 = $(MATMUL_VALIDATE_OBJ) $(MATMUL_CUDA_1_OBJ) $(MATMUL_OBJ)

# Default problem size

SIZE ?= S # Could be S, M or L

# Default target
all: $(TARGET_MATMUL_SEQ) $(TARGET_MATMUL_CUDA_1)

init_tasks: $(BIN_DIR) $(OUTPUT_DIR) GENERATE_INPUT

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

GENERATE_INPUT:
	cd input && sh ./generate.sh

# COMMON

# Compile MATMUL_validate.cu
$(MATMUL_VALIDATE_OBJ): MATMUL_validate.cu | $(BIN_DIR)
	$(CUDACC) $(CUDAFLAGS) -c MATMUL_validate.cu -o $(MATMUL_VALIDATE_OBJ)

# Compile MATMUL.c with nvcc (as C code)
$(MATMUL_OBJ): MATMUL.c | $(BIN_DIR)
	$(CUDACC) -x cu $(CUDAFLAGS) -c MATMUL.c -o $(MATMUL_OBJ)

# SEQ
# Compile MATMUL_seq.c with nvcc (as C code)
$(MATMUL_SEQ_OBJ): MATMUL_seq.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c MATMUL_seq.c -o $(MATMUL_SEQ_OBJ)

# Link all objects into the final binary
$(TARGET_MATMUL_SEQ): $(OBJS_SEQ)
	$(CUDACC) $(OBJS_SEQ) $(LIBS) -o $(TARGET_MATMUL_SEQ)

# CUDA_1
$(MATMUL_CUDA_1_OBJ): MATMUL_cuda_1.cu | $(BIN_DIR)
	$(CUDACC) $(CUDAFLAGS) -c MATMUL_cuda_1.cu -o $(MATMUL_CUDA_1_OBJ)

$(TARGET_MATMUL_CUDA_1): $(OBJS_CUDA_1)
	$(CUDACC) $(OBJS_CUDA_1) $(LIBS) -o $(TARGET_MATMUL_CUDA_1)

# Clean build files
clean_bin:
	rm -rf $(BIN_DIR)/*

clean_output:
	rm -rf $(OUTPUT_DIR)/*

clean_input:
	rm -rf $(INPUT_DIR)/*.in

clean_slurm:
	rm -rf slurm*.out

clean_all:
	make --no-print-directory clean_bin && \
	make --no-print-directory clean_output && \
	make --no-print-directory clean_input && \
	make --no-print-directory clean_slurm

# Full clean, build, and run
init:
	make --no-print-directory clean_all && \
	mkdir -p $(BIN_DIR) && \
	mkdir -p $(OUTPUT_DIR) && \
	cd input && sh ./generate.sh

run_seq:
	make --no-print-directory $(TARGET_MATMUL_SEQ) && \
	sbatch job.sh $(TARGET_MATMUL_SEQ) $(SIZE)

run_cuda_1:
	make --no-print-directory $(TARGET_MATMUL_CUDA_1) && \
	sbatch job.sh $(TARGET_MATMUL_CUDA_1) $(SIZE)
