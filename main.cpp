#include <iostream>
#include <format>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "Functions.h"
#include "CL/cl.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc < 6)
	{
		cerr << std::format("Usage: {} input.jpg output-cpu.jpg output-gpu.jpg a b\n"
			"  Where\n"
			"    [a, b] - pixel base range of the input.jpg\n", argv[0]);
		return -1;
	}

#pragma region Init
	cv::Mat img = imread(argv[1], IMREAD_COLOR);	
	if (img.empty())
	{
		cerr << "Error: Image is not loaded" << endl;
		return -1;
	}

	unsigned char a = atoi(argv[4]);
	unsigned char b = atoi(argv[5]);

	cv::Mat img_cpu = img.clone();
	cv::Mat img_gpu = img.clone();

	cudaEvent_t startCUDA, stopCUDA;
	clock_t startCPU, stopCPU;
	float elapsedTimeCUDA, elapsedTimeCPU;
	cudaEventCreate(&startCUDA);
	cudaEventCreate(&stopCUDA);

	unsigned char* device_img;
	int N = img.rows * img.cols * img.channels();
#pragma endregion

#pragma region CPU
	startCPU = clock();
	adjustBrightnessCPU(img_cpu.ptr<unsigned char>(), img_cpu.rows, img_cpu.cols, a, b);
	stopCPU = clock();

	elapsedTimeCPU = (double)(stopCPU - startCPU) / CLOCKS_PER_SEC;
	cout << "CPU sum time = " << elapsedTimeCPU * 1000 << " ms\n";
#pragma endregion

#pragma region GPU
#ifdef USE_CUDA
	cudaMalloc(&device_img, N * sizeof(unsigned char));
	cudaMemcpy(device_img, img_gpu.ptr<unsigned char>(), N * sizeof(unsigned char), cudaMemcpyHostToDevice);


	const int TPB = 512;
	const int BLK = (img.rows * img.cols * 3 + TPB - 1) / TPB;

	cudaEventRecord(startCUDA, 0);
	adjustBrightnessGPU(BLK, TPB, device_img, img_gpu.rows, img_gpu.cols, a, b);
	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);

	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);
	cudaMemcpy(img_gpu.data, device_img, N * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(device_img);

	cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
	cout << "CUDA memory throughput = " << static_cast<unsigned long long>(N) * sizeof(unsigned char) / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n";
#else
	clock_t startGPU, stopGPU;
	startGPU = clock();
	adjustBrightnessGPUOpenCL(img_gpu.ptr<unsigned char>(), img_gpu.rows, img_gpu.cols, a, b);
	stopGPU = clock();

	float elapsedTimeGPU = (double)(stopGPU - startGPU) / CLOCKS_PER_SEC;
	cout << "GPU time = " << elapsedTimeGPU * 1000 << " ms\n";
#endif
#pragma endregion

#pragma region Results
	imwrite(argv[2], img_cpu);
	imwrite(argv[3], img_gpu);

	int mismatched = 0;
	for (int i = 0; i < N; i++)
	{
		int a = img_cpu.data[i];
		int b = img_gpu.data[i];
		if (abs(a - b) > 3)
		{
			mismatched++;
		}
	}
	cout << "Mismatched: " << mismatched << endl;
#pragma endregion
}