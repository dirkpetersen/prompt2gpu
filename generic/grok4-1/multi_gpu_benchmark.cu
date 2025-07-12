// CUDA benchmark code to utilize multiple GPUs as specified.
// This program allows specifying the number of GPUs via command-line argument.
// It runs a compute and memory-intensive benchmark on each selected GPU,
// utilizing all compute resources and approximately 80% of the GPU memory.
// The benchmark runs for up to 1 minute per GPU thread and starts as quickly as possible.
// Thorough comments are provided throughout.

// Compilation: nvcc -o multi_gpu_benchmark multi_gpu_benchmark.cu -std=c++11
// (Assumes CUDA 12.4, compatible with Ampere architecture sm_86, but code is general.)
// Usage: ./multi_gpu_benchmark <num_gpus>
// Example: ./multi_gpu_benchmark 4  (uses first 4 GPUs)

// Includes for CUDA runtime, standard I/O, threads, timing, and error handling.
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <cstdint>  // For uint64_t

// Error checking macro for CUDA calls.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel function: Performs compute-intensive operations on large arrays to utilize GPU compute and memory.
// This kernel is designed to be both compute-bound (many FLOPs) and memory-bound (large data access).
// Each thread performs repeated multiply-add operations in a loop to keep the GPU busy.
__global__ void benchmarkKernel(float* a, float* b, size_t size, int iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = a[idx];
        for (int i = 0; i < iterations; ++i) {
            val = val * 1.234f + b[idx];  // Simple FMA operation for compute utilization.
            b[idx] = val;  // Write back to ensure memory bandwidth usage.
        }
        a[idx] = val;  // Final write.
    }
}

// Function to run the benchmark on a single GPU.
// This is executed in a separate thread for each GPU to allow concurrent execution.
void runBenchmarkOnDevice(int deviceId, int runtimeSeconds) {
    // Set the current device for this thread.
    CUDA_CHECK(cudaSetDevice(deviceId));

    // Get device properties to determine memory capacity.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    size_t totalMem = prop.totalGlobalMem;
    size_t usableMem = static_cast<size_t>(totalMem * 0.8);  // Target 80% of total memory.

    // Allocate two large float arrays, each taking about 40% of total memory (80% total).
    // Use float for simplicity; size in elements.
    size_t arraySize = (usableMem / 2) / sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, arraySize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, arraySize * sizeof(float)));

    // Initialize arrays with some data to avoid optimizations.
    // Use cudaMemset for quick initialization.
    CUDA_CHECK(cudaMemset(d_a, 0x01, arraySize * sizeof(float)));  // Set to non-zero.
    CUDA_CHECK(cudaMemset(d_b, 0x02, arraySize * sizeof(float)));

    // Kernel launch configuration: Use many blocks and threads to utilize all SMs.
    // Threads per block: 1024 (max for most GPUs).
    // Blocks: Enough to cover the array size and keep GPU occupied.
    int threadsPerBlock = 1024;
    int blocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Number of inner iterations in kernel for compute intensity.
    // Set high to make kernel compute-bound.
    int kernelIterations = 1000;

    // Timing: Run for up to 'runtimeSeconds' seconds.
    auto startTime = std::chrono::steady_clock::now();
    int launchCount = 0;
    while (true) {
        // Launch the kernel.
        benchmarkKernel<<<blocks, threadsPerBlock>>>(d_a, d_b, arraySize, kernelIterations);
        CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors.

        // Synchronize to ensure kernel completes (though for benchmarking, we could overlap).
        CUDA_CHECK(cudaDeviceSynchronize());

        // Check elapsed time.
        auto currentTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = currentTime - startTime;
        if (elapsed.count() >= runtimeSeconds) {
            break;
        }
        launchCount++;
    }

    // Cleanup allocations.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    // Output basic info for this device (to confirm completion).
    std::cout << "Device " << deviceId << " completed " << launchCount << " kernel launches in "
              << runtimeSeconds << " seconds." << std::endl;
}

int main(int argc, char** argv) {
    // Check available GPUs.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    // Diagnostic: Print device info including compute capability.
    std::cout << "Found " << deviceCount << " CUDA devices." << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "Device " << i << ": " << prop.name << " (compute capability " << prop.major << "." << prop.minor << ")" << std::endl;
    }

    int numGpus;
    if (argc == 1) {
        if (deviceCount == 1) {
            numGpus = 1;
        } else {
            std::cerr << "Usage: " << argv[0] << " <num_gpus>" << std::endl;
            std::cerr << "Note: If only one GPU is available, the argument is optional." << std::endl;
            return EXIT_FAILURE;
        }
    } else if (argc == 2) {
        numGpus = std::stoi(argv[1]);
    } else {
        std::cerr << "Usage: " << argv[0] << " <num_gpus>" << std::endl;
        std::cerr << "Note: If only one GPU is available, the argument is optional." << std::endl;
        return EXIT_FAILURE;
    }

    if (numGpus > deviceCount || numGpus < 1) {
        std::cerr << "Invalid number of GPUs: " << numGpus << ". Available: " << deviceCount << std::endl;
        return EXIT_FAILURE;
    }

    // Runtime per GPU: 60 seconds (1 minute).
    const int runtimeSeconds = 60;

    // Create threads for each GPU to run concurrently.
    std::vector<std::thread> threads;
    for (int i = 0; i < numGpus; ++i) {
        threads.emplace_back(runBenchmarkOnDevice, i, runtimeSeconds);
    }

    // Join all threads to wait for completion.
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Benchmark completed on " << numGpus << " GPUs." << std::endl;
    return EXIT_SUCCESS;
}
