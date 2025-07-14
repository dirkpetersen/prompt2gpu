# CUDA GPU Benchmark Makefile
# Compatible with CUDA 12.2+ and compute capabilities sm_86 and sm_90

# Compiler and flags
NVCC = nvcc
CXX = g++

# CUDA architectures for compatibility with RTX 3050 (sm_86) and H200 (sm_90)
CUDA_ARCH = -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90

# Compiler flags
NVCC_FLAGS = -std=c++14 -O3 $(CUDA_ARCH) -Xcompiler -fopenmp
LIBS = -lcublas -lcurand

# Target executable
TARGET = gpu_benchmark
SOURCE = gpu_benchmark.cu

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Install (copy to /usr/local/bin - requires sudo)
install: $(TARGET)
	sudo cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	sudo rm -f /usr/local/bin/$(TARGET)

# Run with default settings (1 GPU)
run: $(TARGET)
	./$(TARGET)

# Run with all available GPUs
run-all: $(TARGET)
	./$(TARGET) $$(nvidia-smi -L | wc -l)

# Display system info
info:
	@echo "System Information:"
	@echo "=================="
	@nvidia-smi
	@echo ""
	@nvcc --version

# Help target
help:
	@echo "CUDA GPU Benchmark Makefile"
	@echo "=========================="
	@echo "Targets:"
	@echo "  all       - Build the benchmark executable"
	@echo "  clean     - Remove build artifacts"
	@echo "  run       - Run benchmark with 1 GPU"
	@echo "  run-all   - Run benchmark with all available GPUs"
	@echo "  install   - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall - Remove from /usr/local/bin (requires sudo)"
	@echo "  info      - Display system and CUDA information"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  ./$(TARGET) [num_gpus]"
	@echo ""
	@echo "Examples:"
	@echo "  ./$(TARGET)     # Use 1 GPU"
	@echo "  ./$(TARGET) 4   # Use 4 GPUs"
	@echo "  ./$(TARGET) 8   # Use 8 GPUs"

.PHONY: all clean install uninstall run run-all info help