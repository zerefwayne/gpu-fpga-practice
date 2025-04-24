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

# Common object files
MATMUL_OBJ            = $(BIN_DIR)/MATMUL.o
MATMUL_VALIDATE_OBJ   = $(BIN_DIR)/MATMUL_validate.o
MATMUL_BENCHMARK_OBJ  = $(BIN_DIR)/MATMUL_benchmark.o

# Code object files
MATMUL_SEQ_OBJ        = $(BIN_DIR)/MATMUL_seq.o
MATMUL_CUDA_1_OBJ 	  = $(BIN_DIR)/MATMUL_cuda_1.o
# NEWFILE: Add your new file here in the same format as the existing one on the previous line

# Common executable targets
TARGET_BENCHMARK = $(BIN_DIR)/MATMUL_benchmark

# Code executable targets
TARGET_MATMUL_SEQ = $(BIN_DIR)/MATMUL_seq
TARGET_MATMUL_CUDA_1 = $(BIN_DIR)/MATMUL_cuda_1
# NEWFILE: Add your new file here in the same format as the existing one in the previous line

# Common object files

# Compile MATMUL_validate.cu
$(MATMUL_VALIDATE_OBJ): MATMUL_validate.cu | $(BIN_DIR)
	$(CUDACC) $(CUDAFLAGS) -c MATMUL_validate.cu -o $(MATMUL_VALIDATE_OBJ)

# Compile MATMUL.c with nvcc (as C code)
$(MATMUL_OBJ): MATMUL.c | $(BIN_DIR)
	$(CUDACC) -x cu $(CUDAFLAGS) -c MATMUL.c -o $(MATMUL_OBJ)

$(MATMUL_BENCHMARK_OBJ): MATMUL_benchmark.c | $(BIN_DIR)
	$(CUDACC) -x cu $(CUDAFLAGS) -c MATMUL_benchmark.c -o $(MATMUL_BENCHMARK_OBJ)

# Object files

# For MATMUL_seq.c
$(MATMUL_SEQ_OBJ): MATMUL_seq.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c MATMUL_seq.c -o $(MATMUL_SEQ_OBJ)

# For MATMUL_cuda_1.c
$(MATMUL_CUDA_1_OBJ): MATMUL_cuda_1.cu | $(BIN_DIR)
	$(CUDACC) $(CUDAFLAGS) -c MATMUL_cuda_1.cu -o $(MATMUL_CUDA_1_OBJ)
# NEWFILE: Add your new file here in the same format as the one on previous 2 lines

init:
	make --no-print-directory clean_all && \
	mkdir -p $(BIN_DIR) && \
	mkdir -p $(OUTPUT_DIR) && \
	cd input && sh ./generate.sh

execute:
	echo "" && \
	echo "Running: $(TARGET_OBJ)" && \
	echo "" && \
	make --no-print-directory $(MATMUL_VALIDATE_OBJ) && \
	make --no-print-directory $(TARGET_OBJ) && \
	make --no-print-directory $(MATMUL_OBJ) && \
	$(CUDACC) $(MATMUL_VALIDATE_OBJ) $(TARGET_OBJ) $(MATMUL_OBJ) $(LIBS) -o $(TARGET_EXECUTABLE) && \
	sbatch job.sh $(TARGET_EXECUTABLE)

run:
ifeq ($(TARGET), seq)
	make --no-print-directory execute TARGET_OBJ=$(MATMUL_SEQ_OBJ) TARGET_EXECUTABLE=$(TARGET_MATMUL_SEQ)
else ifeq ($(TARGET), cuda_1)
	make --no-print-directory execute TARGET_OBJ=$(MATMUL_CUDA_1_OBJ) TARGET_EXECUTABLE=$(TARGET_MATMUL_CUDA_1)
# NEWFILE: Add another for your new file as the previous 2 lines
else
	$(error Invalid TARGET specified (Valid options: seq cuda_1))
# NEWFILE: Append your new file to valid options list
endif

N_BENCHMARKS ?= 10
N_ITERATIONS ?= 10

benchmark:
	echo "" && \
	echo "Benchmarking: $(TARGET_OBJ)" && \
	echo "" && \
	make --no-print-directory $(MATMUL_VALIDATE_OBJ) && \
	make --no-print-directory $(TARGET_OBJ) && \
	make --no-print-directory $(MATMUL_BENCHMARK_OBJ) && \
	$(CUDACC) $(MATMUL_VALIDATE_OBJ) $(TARGET_OBJ) $(MATMUL_BENCHMARK_OBJ) $(LIBS) -o $(TARGET_BENCHMARK) && \
	sbatch benchmark_job.sh $(TARGET_BENCHMARK) $(TARGET_OBJ) $(N_BENCHMARKS) $(N_ITERATIONS)

run_benchmark:
ifeq ($(TARGET), seq)
	make --no-print-directory benchmark TARGET_OBJ=$(MATMUL_SEQ_OBJ) N_BENCHMARKS=$(N_BENCHMARKS) N_ITERATIONS=$(N_ITERATIONS)
else ifeq ($(TARGET), cuda_1)
	make --no-print-directory benchmark TARGET_OBJ=$(MATMUL_CUDA_1_OBJ) N_BENCHMARKS=$(N_BENCHMARKS) N_ITERATIONS=$(N_ITERATIONS)
# NEWFILE: Add another for your new file as the previous 2 lines
else
	$(error Invalid TARGET specified (Valid options: seq cuda_1))
# NEWFILE: Append your new file to valid options list
endif

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
