#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void adjustBrightnessCPU(unsigned char* img, size_t rows, size_t cols, unsigned char a, unsigned char b);
void adjustBrightnessGPU(int BLK, int TPB, unsigned char* img, size_t rows, size_t cols, unsigned char a, unsigned char b);