# Makefile for Multi-GPU CUDA Benchmark

# CUDA installation path (adjust if needed)
CUDA_PATH ?= /usr/local/cuda

# Compiler
NVCC = $(CUDA_PATH)/bin/nvcc

# Base compiler flags
# -O3: Maximum optimization
# -std=c++11: C++11 standard for chrono and threads
NVCCFLAGS_BASE = -O3 -std=c++11

# Architecture-specific flags
ARCH_HOPPER = -arch=sm_90      # H200 GPU (Hopper architecture)
ARCH_AMPERE = -arch=sm_86      # RTX 3050 (Ampere architecture)

# Default architecture (H200)
ARCH ?= $(ARCH_HOPPER)

# Complete compiler flags
NVCCFLAGS = $(NVCCFLAGS_BASE) $(ARCH)

# Target executable
TARGET = multi_gpu_benchmark

# Source file
SOURCE = multi_gpu_benchmark.cu

# Default target (builds for H200)
all: hopper

# Build for H200 (Hopper)
hopper:
	$(NVCC) $(NVCCFLAGS_BASE) $(ARCH_HOPPER) $(SOURCE) -o $(TARGET)
	@echo "Built for H200 (Hopper) GPU"

# Build for RTX 3050 (Ampere)
ampere:
	$(NVCC) $(NVCCFLAGS_BASE) $(ARCH_AMPERE) $(SOURCE) -o $(TARGET)
	@echo "Built for RTX 3050 (Ampere) GPU"

# Build for RTX 3050 (alias)
rtx3050: ampere

# Build with custom architecture
custom:
	$(NVCC) $(NVCCFLAGS) $(SOURCE) -o $(TARGET)
	@echo "Built with custom architecture: $(ARCH)"

# Run with 1 GPU (default)
run: $(TARGET)
	./$(TARGET) 1

# Run with all 8 GPUs
run-all: $(TARGET)
	./$(TARGET) 8

# Run with custom number of GPUs
run-custom: $(TARGET)
	@echo "Usage: make run-custom GPUS=N"
	./$(TARGET) $(GPUS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Show GPU information
gpu-info:
	nvidia-smi

.PHONY: all run run-all run-custom clean gpu-info