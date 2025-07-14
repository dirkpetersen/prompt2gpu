#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Error checking macros
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
        exit(1); \
    } \
} while(0)

// CUDA kernel for compute-intensive operations
__global__ void compute_intensive_kernel(float* data, size_t n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements to maximize compute utilization
    for (size_t i = idx; i < n; i += stride) {
        float value = data[i];
        
        // Perform compute-intensive operations (trigonometric and exponential functions)
        for (int iter = 0; iter < iterations; iter++) {
            value = sinf(value) * cosf(value) + expf(value * 0.001f);
            value = sqrtf(fabsf(value)) + logf(fabsf(value) + 1.0f);
            value = tanhf(value) * 0.5f + value * 0.5f;
        }
        
        data[i] = value;
    }
}

// Matrix multiplication kernel for additional compute load
__global__ void matrix_multiply_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

class GPUBenchmark {
private:
    int num_gpus;
    std::vector<int> gpu_ids;
    std::vector<cudaDeviceProp> gpu_props;
    std::vector<float*> gpu_data;
    std::vector<float*> gpu_matrix_a;
    std::vector<float*> gpu_matrix_b;
    std::vector<float*> gpu_matrix_c;
    std::vector<size_t> data_sizes;
    std::vector<cublasHandle_t> cublas_handles;
    
public:
    GPUBenchmark(int num_gpus_to_use) : num_gpus(num_gpus_to_use) {
        // Initialize CUDA and get device count
        int device_count;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        
        if (num_gpus > device_count) {
            std::cerr << "Requested " << num_gpus << " GPUs but only " << device_count << " available." << std::endl;
            num_gpus = device_count;
        }
        
        // Initialize vectors
        gpu_ids.resize(num_gpus);
        gpu_props.resize(num_gpus);
        gpu_data.resize(num_gpus);
        gpu_matrix_a.resize(num_gpus);
        gpu_matrix_b.resize(num_gpus);
        gpu_matrix_c.resize(num_gpus);
        data_sizes.resize(num_gpus);
        cublas_handles.resize(num_gpus);
        
        // Select GPUs (use first num_gpus devices)
        for (int i = 0; i < num_gpus; i++) {
            gpu_ids[i] = i;
        }
        
        initialize_gpus();
    }
    
    ~GPUBenchmark() {
        cleanup();
    }
    
    void initialize_gpus() {
        std::cout << "Initializing " << num_gpus << " GPU(s) for benchmark..." << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        for (int i = 0; i < num_gpus; i++) {
            CHECK_CUDA(cudaSetDevice(gpu_ids[i]));
            CHECK_CUDA(cudaGetDeviceProperties(&gpu_props[i], gpu_ids[i]));
            
            // Display GPU information
            std::cout << "GPU " << i << " (" << gpu_ids[i] << "): " << gpu_props[i].name << std::endl;
            std::cout << "  Compute Capability: " << gpu_props[i].major << "." << gpu_props[i].minor << std::endl;
            std::cout << "  Total Memory: " << (gpu_props[i].totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;
            std::cout << "  SM Count: " << gpu_props[i].multiProcessorCount << std::endl;
            std::cout << "  Max Threads per Block: " << gpu_props[i].maxThreadsPerBlock << std::endl;
            
            // Calculate memory to use (80% of available memory)
            size_t free_mem, total_mem;
            CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
            size_t target_mem = (size_t)(total_mem * 0.8);
            
            // Allocate memory for compute-intensive operations (70% of target)
            size_t compute_mem = (size_t)(target_mem * 0.7);
            data_sizes[i] = compute_mem / sizeof(float);
            CHECK_CUDA(cudaMalloc(&gpu_data[i], compute_mem));
            
            // Allocate memory for matrix operations (30% of target, split into 3 matrices)
            size_t matrix_mem = (size_t)(target_mem * 0.3) / 3;
            int matrix_size = (int)sqrt(matrix_mem / sizeof(float));
            matrix_size = (matrix_size / 32) * 32; // Align to 32 for better performance
            
            CHECK_CUDA(cudaMalloc(&gpu_matrix_a[i], matrix_size * matrix_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gpu_matrix_b[i], matrix_size * matrix_size * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&gpu_matrix_c[i], matrix_size * matrix_size * sizeof(float)));
            
            std::cout << "  Allocated Memory: " << (target_mem / (1024*1024)) << " MB" << std::endl;
            std::cout << "  Compute Array Size: " << data_sizes[i] << " elements" << std::endl;
            std::cout << "  Matrix Size: " << matrix_size << "x" << matrix_size << std::endl;
            
            // Initialize cuBLAS handle
            CHECK_CUBLAS(cublasCreate(&cublas_handles[i]));
            
            std::cout << std::endl;
        }
    }
    
    void run_benchmark() {
        std::cout << "Starting GPU benchmark (up to 60 seconds)..." << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto benchmark_duration = std::chrono::seconds(60);
        
        // Initialize random data on all GPUs
        for (int i = 0; i < num_gpus; i++) {
            CHECK_CUDA(cudaSetDevice(gpu_ids[i]));
            
            // Create random number generator
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, time(NULL) + i);
            
            // Fill compute array with random data
            curandGenerateUniform(gen, gpu_data[i], data_sizes[i]);
            
            // Fill matrices with random data
            int matrix_size = (int)sqrt((size_t)(gpu_props[i].totalGlobalMem * 0.8 * 0.3) / 3 / sizeof(float));
            matrix_size = (matrix_size / 32) * 32;
            
            curandGenerateUniform(gen, gpu_matrix_a[i], matrix_size * matrix_size);
            curandGenerateUniform(gen, gpu_matrix_b[i], matrix_size * matrix_size);
            
            curandDestroyGenerator(gen);
        }
        
        // Benchmark loop
        int iteration = 0;
        double total_gflops = 0.0;
        
        while (std::chrono::high_resolution_clock::now() - start_time < benchmark_duration) {
            auto iter_start = std::chrono::high_resolution_clock::now();
            
            // Launch kernels on all GPUs
            for (int i = 0; i < num_gpus; i++) {
                CHECK_CUDA(cudaSetDevice(gpu_ids[i]));
                
                // Calculate optimal grid and block dimensions
                int block_size = 256;
                int grid_size = std::min(65535, (int)((data_sizes[i] + block_size - 1) / block_size));
                
                // Launch compute-intensive kernel
                compute_intensive_kernel<<<grid_size, block_size>>>(gpu_data[i], data_sizes[i], 10);
                
                // Launch matrix multiplication kernel
                int matrix_size = (int)sqrt((size_t)(gpu_props[i].totalGlobalMem * 0.8 * 0.3) / 3 / sizeof(float));
                matrix_size = (matrix_size / 32) * 32;
                
                dim3 block_dim(16, 16);
                dim3 grid_dim((matrix_size + block_dim.x - 1) / block_dim.x, 
                             (matrix_size + block_dim.y - 1) / block_dim.y);
                
                if (matrix_size > 0) {
                    matrix_multiply_kernel<<<grid_dim, block_dim>>>(gpu_matrix_a[i], gpu_matrix_b[i], gpu_matrix_c[i], matrix_size);
                }
                
                // Additional cuBLAS operation for more compute
                if (matrix_size > 0) {
                    float alpha = 1.0f, beta = 0.0f;
                    CHECK_CUBLAS(cublasSgemm(cublas_handles[i], CUBLAS_OP_N, CUBLAS_OP_N,
                                           matrix_size, matrix_size, matrix_size,
                                           &alpha, gpu_matrix_a[i], matrix_size,
                                                   gpu_matrix_b[i], matrix_size,
                                           &beta, gpu_matrix_c[i], matrix_size));
                }
            }
            
            // Synchronize all GPUs
            for (int i = 0; i < num_gpus; i++) {
                CHECK_CUDA(cudaSetDevice(gpu_ids[i]));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            
            auto iter_end = std::chrono::high_resolution_clock::now();
            auto iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start);
            
            // Estimate GFLOPS (rough calculation)
            double ops_per_gpu = data_sizes[0] * 20.0; // Approximate ops per element in kernel
            double total_ops = ops_per_gpu * num_gpus;
            double gflops = (total_ops / 1e9) / (iter_duration.count() / 1000.0);
            total_gflops += gflops;
            
            iteration++;
            
            if (iteration % 10 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
                std::cout << "Iteration " << std::setw(4) << iteration 
                         << " | Elapsed: " << std::setw(2) << elapsed.count() << "s"
                         << " | GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Benchmark completed!" << std::endl;
        std::cout << "Total iterations: " << iteration << std::endl;
        std::cout << "Total time: " << (total_duration.count() / 1000.0) << " seconds" << std::endl;
        std::cout << "Average GFLOPS: " << std::fixed << std::setprecision(2) << (total_gflops / iteration) << std::endl;
        std::cout << "GPUs utilized: " << num_gpus << std::endl;
    }
    
private:
    void cleanup() {
        for (int i = 0; i < num_gpus; i++) {
            CHECK_CUDA(cudaSetDevice(gpu_ids[i]));
            
            if (gpu_data[i]) cudaFree(gpu_data[i]);
            if (gpu_matrix_a[i]) cudaFree(gpu_matrix_a[i]);
            if (gpu_matrix_b[i]) cudaFree(gpu_matrix_b[i]);
            if (gpu_matrix_c[i]) cudaFree(gpu_matrix_c[i]);
            
            if (cublas_handles[i]) cublasDestroy(cublas_handles[i]);
        }
    }
};

int main(int argc, char* argv[]) {
    int num_gpus = 1;
    
    // Parse command line arguments
    if (argc > 1) {
        num_gpus = std::atoi(argv[1]);
        if (num_gpus <= 0) {
            std::cerr << "Invalid number of GPUs: " << num_gpus << std::endl;
            return 1;
        }
    }
    
    std::cout << "CUDA GPU Benchmark Tool" << std::endl;
    std::cout << "Requested GPUs: " << num_gpus << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    try {
        GPUBenchmark benchmark(num_gpus);
        benchmark.run_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}