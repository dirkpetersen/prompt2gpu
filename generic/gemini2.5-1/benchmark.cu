#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cmath>

// CUDA runtime
#include <cuda_runtime.h>

/**
 * @brief Macro to wrap CUDA calls and check for errors.
 * 
 * If a CUDA call fails, it prints the error message with file and line number,
 * and then terminates the program.
 */
#define CUDA_CHECK(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

/**
 * @brief A compute-intensive kernel to keep the GPU busy.
 * 
 * This kernel performs a series of floating-point operations on each element
 * of the input array. The goal is to saturate the GPU's compute units.
 * 
 * @param data A pointer to the device memory array.
 * @param size The total number of elements in the array.
 */
__global__ void benchmark_kernel(float* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        // Perform some arbitrary floating point operations in a loop to keep the core busy.
        // Using transcendental functions to ensure significant computational work.
        for (int i = 0; i < 1000; ++i) {
            val = sinf(val) * 0.5f + cosf(val) * 0.5f;
        }
        data[idx] = val;
    }
}

/**
 * @brief The worker function that runs the benchmark on a single designated GPU.
 * 
 * This function is executed in a separate thread for each GPU being benchmarked.
 * It allocates 80% of the GPU's memory, and continuously launches the benchmark_kernel
 * until the main thread signals it to stop.
 * 
 * @param device_id The ID of the GPU for this worker to use.
 * @param keep_running An atomic boolean flag, controlled by the main thread, to signal when to stop.
 */
void gpu_worker(int device_id, std::atomic<bool>& keep_running) {
    try {
        // Set the current CUDA device for this thread.
        CUDA_CHECK(cudaSetDevice(device_id));

        // Get device properties to display its name and get total memory.
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        printf("GPU %d: %s\n", device_id, prop.name);

        // Calculate 80% of the total memory capacity of the GPU.
        size_t memory_to_use = static_cast<size_t>(prop.totalGlobalMem * 0.8);
        size_t num_elements = memory_to_use / sizeof(float);

        // Allocate the memory on the GPU.
        float* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, memory_to_use));

        // Initialize allocated memory to prevent operating on uninitialized data.
        // A simple memset is sufficient.
        CUDA_CHECK(cudaMemset(d_data, 0, memory_to_use));

        // Configure kernel launch parameters to saturate the GPU.
        // A common block size is 256 or 512 threads.
        int threads_per_block = 256;
        // Calculate the number of blocks needed to cover all elements.
        int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        
        printf("GPU %d: Allocating %.2f GB of memory. Launching kernel with %d blocks and %d threads per block.\n",
               device_id, memory_to_use / (1024.0 * 1024.0 * 1024.0), blocks_per_grid, threads_per_block);

        // Main benchmark loop. Continues as long as keep_running is true.
        while (keep_running) {
            benchmark_kernel<<<blocks_per_grid, threads_per_block>>>(d_data, num_elements);
            // Check for any errors during kernel launch.
            CUDA_CHECK(cudaGetLastError());
            // Synchronize the device to ensure one kernel completes before starting another
            // and to allow this thread to check the keep_running flag periodically.
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Free the allocated GPU memory.
        CUDA_CHECK(cudaFree(d_data));
        printf("GPU %d: Benchmark finished. Releasing resources.\n", device_id);

    } catch (const std::exception& e) {
        fprintf(stderr, "An error occurred on GPU %d: %s\n", device_id, e.what());
    }
}

/**
 * @brief Main entry point of the application.
 * 
 * Parses command-line arguments for number of GPUs and duration.
 * Spawns worker threads for each GPU and manages the benchmark duration.
 */
int main(int argc, char** argv) {
    int num_gpus_to_use = 0;
    int duration_seconds = 60; // Default duration of 60 seconds.

    // Parse command-line arguments:
    // argv[1]: number of GPUs to use.
    // argv[2]: duration of the benchmark in seconds.
    if (argc > 1) {
        try {
            num_gpus_to_use = std::stoi(argv[1]);
        } catch (...) {
            fprintf(stderr, "Invalid number of GPUs specified. Please provide an integer.\n");
            return 1;
        }
    }
    if (argc > 2) {
        try {
            duration_seconds = std::stoi(argv[2]);
        } catch (...) {
            fprintf(stderr, "Invalid duration specified. Please provide an integer.\n");
            return 1;
        }
    }

    // Get the total number of CUDA-capable devices.
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-enabled GPUs found.\n");
        return 1;
    }

    // If the user-specified number of GPUs is invalid or not provided, use all available GPUs.
    if (num_gpus_to_use <= 0 || num_gpus_to_use > device_count) {
        printf("Number of GPUs not specified or invalid. Using all %d available GPU(s).\n", device_count);
        num_gpus_to_use = device_count;
    }

    printf("Starting benchmark on %d GPU(s) for %d seconds.\n", num_gpus_to_use, duration_seconds);

    std::vector<std::thread> gpu_threads;
    std::atomic<bool> keep_running(true);

    // Launch a worker thread for each GPU to be benchmarked.
    for (int i = 0; i < num_gpus_to_use; ++i) {
        gpu_threads.emplace_back(gpu_worker, i, std::ref(keep_running));
    }

    // Let the benchmark run for the specified duration.
    std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));

    // Signal worker threads to stop.
    keep_running = false;
    printf("Time limit reached. Signaling %zu worker thread(s) to stop...\n", gpu_threads.size());

    // Wait for all threads to finish their current work and clean up.
    for (auto& t : gpu_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    printf("Benchmark complete.\n");
    return 0;
}
