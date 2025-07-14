/*
 * Multi-GPU CUDA Benchmark Program
 * 
 * This program runs a compute-intensive benchmark on multiple NVIDIA GPUs
 * Features:
 * - Configurable number of GPUs via command line argument
 * - Detects and displays GPU type
 * - Allocates 80% of available GPU memory
 * - Runs compute-intensive operations for approximately 1 minute
 * - Displays performance metrics
 * 
 * Compile: nvcc -O3 -arch=sm_90 multi_gpu_benchmark.cu -o multi_gpu_benchmark
 * Run: ./multi_gpu_benchmark [num_gpus]
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <thread>

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Compute-intensive kernel that performs matrix operations
// This kernel performs repeated floating-point operations to stress the GPU
__global__ void benchmark_kernel(float* data, size_t n_elements, int iterations) {
    // Calculate global thread ID
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop to handle more elements than threads
    size_t stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements with complex operations
    for (size_t i = idx; i < n_elements; i += stride) {
        float value = data[i];
        
        // Perform compute-intensive operations
        for (int iter = 0; iter < iterations; iter++) {
            // Mix of transcendental and arithmetic operations
            value = sinf(value) * cosf(value) + sqrtf(fabsf(value));
            value = value * 1.1f - 0.1f;
            value = expf(-value * value) + logf(fabsf(value) + 1.0f);
            value = fmaf(value, 2.0f, 1.0f); // Fused multiply-add
            value = tanhf(value) * atanf(value);
        }
        
        // Write result back to global memory
        data[i] = value;
    }
}

// Structure to hold GPU information and allocated resources
struct GPUContext {
    int device_id;
    char device_name[256];
    size_t total_memory;
    size_t allocated_memory;
    float* d_data;
    cudaStream_t stream;
};

// Function to display GPU properties
void display_gpu_info(const GPUContext& ctx) {
    printf("\nGPU %d: %s\n", ctx.device_id, ctx.device_name);
    printf("  Total Memory: %.2f GB\n", ctx.total_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Allocated Memory: %.2f GB (80%%)\n", ctx.allocated_memory / (1024.0 * 1024.0 * 1024.0));
}

// Function to run benchmark on a single GPU
void run_gpu_benchmark(GPUContext& ctx, int duration_seconds) {
    // Set the current device
    CHECK_CUDA(cudaSetDevice(ctx.device_id));
    
    // Calculate kernel launch parameters
    int block_size = 256;  // Threads per block
    size_t n_elements = ctx.allocated_memory / sizeof(float);
    int grid_size = (n_elements + block_size - 1) / block_size;
    
    // Limit grid size to avoid exceeding maximum
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    
    printf("GPU %d: Launching kernel with %d blocks of %d threads\n", 
           ctx.device_id, grid_size, block_size);
    
    // Initialize data with random values
    std::vector<float> h_init(n_elements);
    for (size_t i = 0; i < n_elements; i++) {
        h_init[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Copy initial data to GPU
    CHECK_CUDA(cudaMemcpyAsync(ctx.d_data, h_init.data(), 
                               ctx.allocated_memory, 
                               cudaMemcpyHostToDevice, ctx.stream));
    
    // Synchronize before starting timing
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = start_time;
    int kernel_launches = 0;
    
    // Run kernels for the specified duration
    while (std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() < duration_seconds) {
        // Launch kernel with varying iteration count to maintain load
        int iterations = 50 + (kernel_launches % 50);  // Vary between 50-100 iterations
        
        benchmark_kernel<<<grid_size, block_size, 0, ctx.stream>>>(
            ctx.d_data, n_elements, iterations);
        
        kernel_launches++;
        
        // Check for errors periodically
        if (kernel_launches % 100 == 0) {
            CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
            current_time = std::chrono::high_resolution_clock::now();
        } else {
            current_time = std::chrono::high_resolution_clock::now();
        }
    }
    
    // Final synchronization
    CHECK_CUDA(cudaStreamSynchronize(ctx.stream));
    
    // Calculate elapsed time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    printf("GPU %d: Completed %d kernel launches in %.2f seconds\n", 
           ctx.device_id, kernel_launches, elapsed.count() / 1000.0);
    
    // Calculate approximate FLOPS (rough estimate)
    // Each iteration in kernel performs ~10 floating-point operations
    double total_operations = (double)kernel_launches * n_elements * 75.0 * 10.0;
    double gflops = total_operations / (elapsed.count() / 1000.0) / 1e9;
    
    printf("GPU %d: Approximate performance: %.2f GFLOPS\n", ctx.device_id, gflops);
}

// Thread function to run benchmark on a specific GPU
void gpu_thread_func(GPUContext& ctx, int duration_seconds) {
    run_gpu_benchmark(ctx, duration_seconds);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int num_gpus_to_use = 1;  // Default to 1 GPU
    if (argc > 1) {
        num_gpus_to_use = atoi(argv[1]);
    }
    
    // Get total number of available GPUs
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable GPUs found!\n");
        return 1;
    }
    
    printf("Found %d CUDA-capable GPU(s)\n", device_count);
    
    // Validate requested number of GPUs
    if (num_gpus_to_use < 1 || num_gpus_to_use > device_count) {
        fprintf(stderr, "Invalid number of GPUs requested: %d (available: %d)\n", 
                num_gpus_to_use, device_count);
        return 1;
    }
    
    printf("Using %d GPU(s) for benchmark\n", num_gpus_to_use);
    
    // Create GPU contexts
    std::vector<GPUContext> gpu_contexts(num_gpus_to_use);
    
    // Initialize each GPU
    for (int i = 0; i < num_gpus_to_use; i++) {
        GPUContext& ctx = gpu_contexts[i];
        ctx.device_id = i;
        
        // Set device and get properties
        CHECK_CUDA(cudaSetDevice(i));
        
        cudaDeviceProp props;
        CHECK_CUDA(cudaGetDeviceProperties(&props, i));
        
        // Store device information
        snprintf(ctx.device_name, sizeof(ctx.device_name), "%s", props.name);
        ctx.total_memory = props.totalGlobalMem;
        ctx.allocated_memory = (size_t)(ctx.total_memory * 0.8);  // 80% of total memory
        
        // Display GPU information
        display_gpu_info(ctx);
        
        // Create stream for this GPU
        CHECK_CUDA(cudaStreamCreate(&ctx.stream));
        
        // Allocate GPU memory
        CHECK_CUDA(cudaMalloc(&ctx.d_data, ctx.allocated_memory));
        
        printf("GPU %d: Memory allocation successful\n", i);
    }
    
    printf("\nStarting benchmark for 60 seconds...\n");
    printf("========================================\n");
    
    // Create threads for each GPU
    std::vector<std::thread> gpu_threads;
    
    // Launch benchmark on each GPU in a separate thread
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_gpus_to_use; i++) {
        gpu_threads.emplace_back(gpu_thread_func, std::ref(gpu_contexts[i]), 60);
    }
    
    // Wait for all GPU threads to complete
    for (auto& thread : gpu_threads) {
        thread.join();
    }
    
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(benchmark_end - benchmark_start);
    
    printf("\n========================================\n");
    printf("Benchmark completed in %ld seconds\n", total_elapsed.count());
    
    // Cleanup
    for (auto& ctx : gpu_contexts) {
        CHECK_CUDA(cudaSetDevice(ctx.device_id));
        CHECK_CUDA(cudaFree(ctx.d_data));
        CHECK_CUDA(cudaStreamDestroy(ctx.stream));
    }
    
    printf("Cleanup completed successfully\n");
    
    return 0;
}