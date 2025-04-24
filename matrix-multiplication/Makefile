# Compilers
CC      := gcc
CUDACC  := nvcc

# Flags
CFLAGS      := -O3 -Wall
CUDAFLAGS   := -O3
LIBS        := -lm -lcublas

# Directories
BIN_DIR     := bin
OUTPUT_DIR  := output
INPUT_DIR   := input

# Support sources (always included)
VALIDATE_SRC   := MATMUL_validate.cu
SUPPORT_SRC    := MATMUL.c
BENCH_SRC      := MATMUL_benchmark.c

# Discover all candidate implementation sources
CU_SRCS       := $(wildcard MATMUL_*.cu)
IMPL_CU_SRCS  := $(filter-out $(VALIDATE_SRC),$(CU_SRCS))

C_SRCS        := $(wildcard MATMUL_*.c)
IMPL_C_SRCS   := $(filter-out $(SUPPORT_SRC) $(BENCH_SRC),$(C_SRCS))

# Names like "seq", "cuda_1", etc.
IMPL_NAMES    := $(sort \
                    $(patsubst MATMUL_%.cu,%,$(IMPL_CU_SRCS)) \
                    $(patsubst MATMUL_%.c,%, $(IMPL_C_SRCS)) \
                 )

# Build lists
SUPPORT_OBJS  := $(BIN_DIR)/MATMUL_validate.o $(BIN_DIR)/MATMUL.o
BENCH_OBJ     := $(BIN_DIR)/MATMUL_benchmark.o
IMPL_OBJS     := $(IMPL_NAMES:%=$(BIN_DIR)/MATMUL_%.o)
IMPL_EXES     := $(IMPL_NAMES:%=$(BIN_DIR)/MATMUL_%)

# Runtime parameters (override on CLI)
N_BENCHMARKS   ?= 10
N_ITERATIONS   ?= 10

.PHONY: all init clean clean_all run benchmark

# ——————————————————————————————————————————————
# Default: build every implementation
all: $(BIN_DIR) $(IMPL_EXES)

# Ensure bin directory exists
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Compile any .cu → .o
$(BIN_DIR)/%.o: %.cu | $(BIN_DIR)
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

# Compile any .c → .o
$(BIN_DIR)/%.o: %.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link each implementation into its executable
$(BIN_DIR)/MATMUL_%: $(BIN_DIR)/MATMUL_%.o $(SUPPORT_OBJS)
	$(CUDACC) $^ $(LIBS) -o $@

# Explicit support-object rules
$(BIN_DIR)/MATMUL_validate.o: MATMUL_validate.cu | $(BIN_DIR)
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(BIN_DIR)/MATMUL.o: MATMUL.c | $(BIN_DIR)
	$(CUDACC) -x cu $(CUDAFLAGS) -c $< -o $@

$(BIN_DIR)/MATMUL_benchmark.o: MATMUL_benchmark.c | $(BIN_DIR)
	$(CUDACC) -x cu $(CUDAFLAGS) -c $< -o $@

# ——————————————————————————————————————————————
# run: build & submit SLURM job for one variant
# Usage: make run TARGET=<name> INPUT_FILE=<input file path>
run:
	@if [ -z "$(TARGET)" ]; then \
	  echo "Error: please set TARGET, e.g. 'make run TARGET=seq'"; \
	  exit 1; \
	fi
	@if [ -z "$(INPUT_FILE)" ]; then \
	  echo "Error: please set INPUT_FILE, e.g. 'make run TARGET=seq INPUT_FILE=path/to/input'"; \
	  exit 1; \
	fi
	@if [ ! -f "$(INPUT_FILE)" ]; then \
	  echo "Error: input file '$(INPUT_FILE)' does not exist."; \
	  exit 1; \
	fi
	@$(MAKE) --no-print-directory $(BIN_DIR)/MATMUL_$(TARGET)
	@if [ ! -x $(BIN_DIR)/MATMUL_$(TARGET) ]; then \
	  echo "Error: binary '$(BIN_DIR)/MATMUL_$(TARGET)' not found."; \
	  echo "  Build it first with 'make $(BIN_DIR)/MATMUL_$(TARGET)'"; \
	  exit 1; \
	fi
	sbatch run_job.sh $(BIN_DIR)/MATMUL_$(TARGET) $(INPUT_FILE)

# ——————————————————————————————————————————————
# benchmark: build & submit SLURM benchmark for one variant
# Usage: make benchmark TARGET=<name> [N_BENCHMARKS=<n>] [N_ITERATIONS=<m>]
benchmark:
	@if [ -z "$(TARGET)" ]; then \
	  echo "Error: please set TARGET, e.g. 'make benchmark TARGET=seq'"; \
	  exit 1; \
	fi
	@if [ ! -f MATMUL_$(TARGET).cu ] && [ ! -f MATMUL_$(TARGET).c ]; then \
	  echo "Error: no source 'MATMUL_$(TARGET).cu' or '.c' found."; \
	  exit 1; \
	fi

	# ensure all three objects are built
	$(MAKE) --no-print-directory $(BIN_DIR)/MATMUL_validate.o
	$(MAKE) --no-print-directory $(BIN_DIR)/MATMUL_$(TARGET).o
	$(MAKE) --no-print-directory $(BIN_DIR)/MATMUL_benchmark.o

	# now link them together
	$(CUDACC) \
	  $(BIN_DIR)/MATMUL_validate.o \
	  $(BIN_DIR)/MATMUL_$(TARGET).o \
	  $(BIN_DIR)/MATMUL_benchmark.o \
	  $(LIBS) -o $(BIN_DIR)/MATMUL_benchmark

	# submit
	sbatch benchmark_job.sh \
	  $(BIN_DIR)/MATMUL_benchmark \
	  $(TARGET) $(N_BENCHMARKS) $(N_ITERATIONS)

# ——————————————————————————————————————————————
init:
	make --no-print-directory clean_all && \
	mkdir -p $(BIN_DIR) && \
	mkdir -p $(OUTPUT_DIR) && \
	cd input && sh ./generate.sh

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
