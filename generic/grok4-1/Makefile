all: multi_gpu_benchmark

multi_gpu_benchmark: multi_gpu_benchmark.cu
	nvcc -o multi_gpu_benchmark multi_gpu_benchmark.cu -std=c++11

clean:
	rm -f multi_gpu_benchmark
