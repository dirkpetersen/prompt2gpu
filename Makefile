# Compiler
NVCC = nvcc

# Compiler flags
# -O3 for optimization
# -std=c++17 to use modern C++ features like std::atomic and std::thread
NVCCFLAGS = -O3 -std=c++17

# Target architectures
# For RTX 3050 (Ampere architecture)
ARCH_SM86 = -gencode arch=compute_86,code=sm_86
# For H200 (Hopper architecture)
ARCH_SM90 = -gencode arch=compute_90,code=sm_90

# Combine architectures for a fat binary. This allows the same executable
# to run on both development and production machines.
ARCHS = $(ARCH_SM86) $(ARCH_SM90)

# Target executable name
TARGET = benchmark

# Source files
SOURCES = benchmark.cu

# Default target to build the executable
all: $(TARGET)

# Rule to link the final executable
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) $(ARCHS) -o $@ $(SOURCES)

# Rule to clean up build artifacts
clean:
	rm -f $(TARGET)
