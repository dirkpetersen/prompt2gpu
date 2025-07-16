# CUDA Multi-GPU Benchmark Makefile
# Compatible with CUDA 12.2+ and architectures sm_86, sm_90

# Compiler and flags
NVCC = nvcc
CXX = g++

# CUDA architectures for both RTX 3050 (sm_86) and H200 (sm_90)
CUDA_ARCH = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90

# Compiler flags
NVCC_FLAGS = -O3 -std=c++14 $(CUDA_ARCH) -Xcompiler -fopenmp
LDFLAGS = -lcuda -lcudart -lgomp

# Source and target
SOURCE = cuda_benchmark.cu
TARGET = cuda_benchmark

# Default target
all: $(TARGET)

# Build the benchmark
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run benchmark with default settings (1 GPU, 60 seconds)
run: $(TARGET)
	./$(TARGET)

# Run benchmark with specified number of GPUs
run-multi: $(TARGET)
	./$(TARGET) 4 60

# Run quick test (10 seconds)
test: $(TARGET)
	./$(TARGET) 1 10

# Show GPU information
gpu-info:
	nvidia-smi

# Check CUDA installation
cuda-check:
	nvcc --version
	nvidia-smi

# Installation help
help:
	@echo "CUDA Multi-GPU Benchmark Makefile"
	@echo "=================================="
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the benchmark (default)"
	@echo "  clean      - Remove build artifacts"
	@echo "  run        - Run benchmark with 1 GPU for 60 seconds"
	@echo "  run-multi  - Run benchmark with 4 GPUs for 60 seconds"
	@echo "  test       - Quick test with 1 GPU for 10 seconds"
	@echo "  gpu-info   - Show GPU information"
	@echo "  cuda-check - Check CUDA installation"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make                    # Build"
	@echo "  make run               # Run with defaults"
	@echo "  ./cuda_benchmark       # 1 GPU, 60 seconds"
	@echo "  ./cuda_benchmark 4     # 4 GPUs, 60 seconds"
	@echo "  ./cuda_benchmark 2 30  # 2 GPUs, 30 seconds"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA 12.2+ (compatible with 12.4)"
	@echo "  - GCC with OpenMP support"
	@echo "  - GPU with compute capability 8.6+ (RTX 3050) or 9.0+ (H200)"

.PHONY: all clean run run-multi test gpu-info cuda-check help