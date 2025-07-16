#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <memory>
#include <iomanip>

// CUDA kernel for compute-intensive matrix multiplication benchmark
// Each thread computes multiple dot products to maximize compute utilization
__global__ void compute_benchmark_kernel(float* a, float* b, float* c, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements to maximize compute workload
    for (int i = idx; i < size; i += stride) {
        float sum = 0.0f;
        
        // Perform multiple iterations of compute-intensive operations
        for (int iter = 0; iter < iterations; iter++) {
            // Matrix-like computation with transcendental functions
            float val_a = a[i % size];
            float val_b = b[i % size];
            
            // Compute-intensive operations: trigonometric and exponential functions
            sum += sinf(val_a) * cosf(val_b) + expf(val_a * 0.001f) * logf(val_b + 1.0f);
            sum += sqrtf(val_a * val_a + val_b * val_b);
            sum += powf(val_a, 1.5f) + powf(val_b, 2.3f);
        }
        
        c[i] = sum;
    }
}

// Function to get GPU properties and display information
void displayGPUInfo(int deviceId) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    std::cout << "GPU " << deviceId << ": " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << std::endl;
}

// Function to calculate optimal grid and block dimensions
void calculateOptimalDimensions(int deviceId, int dataSize, dim3& grid, dim3& block) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    // Use optimal block size based on GPU architecture
    int blockSize = 256; // Good default for most modern GPUs
    if (prop.major >= 7) { // Volta/Turing/Ampere/Ada/Hopper
        blockSize = 512;
    }
    
    // Calculate grid size to cover all data elements
    int gridSize = (dataSize + blockSize - 1) / blockSize;
    
    // Limit grid size to avoid excessive overhead
    gridSize = std::min(gridSize, prop.multiProcessorCount * 8);
    
    block = dim3(blockSize);
    grid = dim3(gridSize);
}

// Main benchmark function for a single GPU
void runGPUBenchmark(int deviceId, size_t memorySize, int durationSeconds) {
    // Set the current device
    cudaSetDevice(deviceId);
    
    // Display GPU information
    displayGPUInfo(deviceId);
    
    // Calculate array size based on available memory (80% utilization)
    // Each element is 4 bytes (float), we need 3 arrays (a, b, c)
    size_t elementsPerArray = memorySize / (3 * sizeof(float));
    size_t totalElements = elementsPerArray;
    
    std::cout << "Allocating " << (memorySize / (1024*1024)) << " MB per GPU" << std::endl;
    std::cout << "Array size: " << totalElements << " elements" << std::endl;
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, totalElements * sizeof(float));
    cudaMalloc(&d_b, totalElements * sizeof(float));
    cudaMalloc(&d_c, totalElements * sizeof(float));
    
    // Allocate and initialize host memory
    std::vector<float> h_a(totalElements);
    std::vector<float> h_b(totalElements);
    
    // Initialize arrays with random-like values
    for (size_t i = 0; i < totalElements; i++) {
        h_a[i] = static_cast<float>(i % 1000) / 1000.0f + 1.0f;
        h_b[i] = static_cast<float>((i * 7) % 1000) / 1000.0f + 1.0f;
    }
    
    // Copy data to device
    cudaMemcpy(d_a, h_a.data(), totalElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), totalElements * sizeof(float), cudaMemcpyHostToDevice);
    
    // Calculate optimal kernel launch parameters
    dim3 grid, block;
    calculateOptimalDimensions(deviceId, totalElements, grid, block);
    
    std::cout << "Kernel config: Grid(" << grid.x << "), Block(" << block.x << ")" << std::endl;
    
    // Warm-up kernel launch
    compute_benchmark_kernel<<<grid, block>>>(d_a, d_b, d_c, totalElements, 10);
    cudaDeviceSynchronize();
    
    // Benchmark timing
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime = startTime + std::chrono::seconds(durationSeconds);
    
    int iterations = 0;
    int computeIterations = 100; // Number of compute iterations per kernel launch
    
    std::cout << "Running benchmark for " << durationSeconds << " seconds..." << std::endl;
    
    // Main benchmark loop
    while (std::chrono::high_resolution_clock::now() < endTime) {
        // Launch compute-intensive kernel
        compute_benchmark_kernel<<<grid, block>>>(d_a, d_b, d_c, totalElements, computeIterations);
        iterations++;
        
        // Synchronize every few iterations to check timing
        if (iterations % 10 == 0) {
            cudaDeviceSynchronize();
        }
    }
    
    // Final synchronization
    cudaDeviceSynchronize();
    
    auto actualEndTime = std::chrono::high_resolution_clock::now();
    auto actualDuration = std::chrono::duration_cast<std::chrono::milliseconds>(actualEndTime - startTime);
    
    // Calculate performance metrics
    double totalOperations = static_cast<double>(iterations) * totalElements * computeIterations * 8; // 8 ops per iteration
    double gflops = (totalOperations / 1e9) / (actualDuration.count() / 1000.0);
    
    std::cout << "GPU " << deviceId << " Benchmark Results:" << std::endl;
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Actual Duration: " << actualDuration.count() << " ms" << std::endl;
    std::cout << "  Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int numGPUs = 1;
    int duration = 60; // Default 60 seconds
    
    if (argc > 1) {
        numGPUs = std::atoi(argv[1]);
    }
    if (argc > 2) {
        duration = std::atoi(argv[2]);
    }
    
    // Validate duration (max 60 seconds as specified)
    if (duration > 60) {
        duration = 60;
        std::cout << "Duration capped at 60 seconds as specified." << std::endl;
    }
    
    std::cout << "CUDA Multi-GPU Benchmark" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Check available GPUs
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Available GPUs: " << deviceCount << std::endl;
    
    if (numGPUs > deviceCount) {
        std::cout << "Requested " << numGPUs << " GPUs, but only " << deviceCount << " available." << std::endl;
        numGPUs = deviceCount;
    }
    
    std::cout << "Using " << numGPUs << " GPU(s)" << std::endl;
    std::cout << "Benchmark duration: " << duration << " seconds" << std::endl;
    std::cout << std::endl;
    
    // Get memory information for each GPU and calculate 80% utilization
    std::vector<size_t> memoryPerGPU(numGPUs);
    
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        // Use 80% of available memory (not total memory, to account for system usage)
        memoryPerGPU[i] = static_cast<size_t>(free_mem * 0.8);
        
        std::cout << "GPU " << i << " - Total: " << (total_mem / (1024*1024)) << " MB, "
                  << "Free: " << (free_mem / (1024*1024)) << " MB, "
                  << "Using: " << (memoryPerGPU[i] / (1024*1024)) << " MB (80%)" << std::endl;
    }
    std::cout << std::endl;
    
    // Create streams for each GPU to enable concurrent execution
    std::vector<cudaStream_t> streams(numGPUs);
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
    }
    
    auto globalStartTime = std::chrono::high_resolution_clock::now();
    
    // Launch benchmark on all GPUs concurrently using OpenMP-style approach
    #pragma omp parallel for
    for (int i = 0; i < numGPUs; i++) {
        runGPUBenchmark(i, memoryPerGPU[i], duration);
    }
    
    auto globalEndTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(globalEndTime - globalStartTime);
    
    std::cout << "Total benchmark completed in " << totalDuration.count() << " ms" << std::endl;
    
    // Cleanup streams
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaStreamDestroy(streams[i]);
    }
    
    // Reset all devices
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }
    
    return 0;
}