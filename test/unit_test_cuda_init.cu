#include "cuda_runtime.h"

void initialize_cuda_runtime(void)
{
    cudaSetDevice(0);
    cudaFree(0);
}