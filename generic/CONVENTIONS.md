I have a production Nvidia DGX Machine with 8xH200 which has Cuda 12.4 installed, Driver Version: 550.144.03. The OS is "Ubuntu 22.04.5 LTS“, NVCC Architecture: sm_90. Then I have a development machine with a single RTX 3050, Driver Version: 537.13, CUDA Version: 12.2, NVCC Architecture: sm_86

Write me some Cuda code that allows me to specify the number of GPUs to use. This code should run some sort of benchmark and utilize all compute as well as 80% of the memory capacity of the GPUs selected. It must also display the type of GPU installed. It should run up to one minute and start as quickly as possible. Please comment the code you write thoroughly. I must be able to compile the code on the DGX as well as the machine with the RTX3050

