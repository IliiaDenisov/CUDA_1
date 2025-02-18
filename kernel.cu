#include "Functions.h"

__device__ unsigned char adjustChannel(unsigned char channelValue, unsigned char a, unsigned char b) 
{
	return static_cast<unsigned char>(trunc((static_cast<double>(channelValue - a) / (b - a)) * 255.0));
}

__global__ void adjustBrightnessGPU_kernel(unsigned char* img, size_t rows, size_t cols, unsigned char a, unsigned char b)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < (rows * cols * 3)) { 
        img[id] = static_cast<unsigned char>((img[id] - a) * 255 / (b - a));
    }
}

void adjustBrightnessGPU(int BLK, int TPB, unsigned char* img, size_t rows, size_t cols, unsigned char a, unsigned char b)
{
    adjustBrightnessGPU_kernel << <BLK, TPB >> > (img, rows, cols, a, b);
}