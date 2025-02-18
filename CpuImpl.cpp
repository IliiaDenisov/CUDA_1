#include "Functions.h"
#include "CL/cl.hpp"
#include <vector>

using namespace std;

void adjustBrightnessCPU(unsigned char* img, size_t rows, size_t cols, unsigned char a, unsigned char b)
{
	auto adjustChannel = [a, b](unsigned char& channelValue) {
		channelValue = static_cast<unsigned char>((channelValue - a) * 255 / (b - a));
	};

	for (size_t i = 0; i < (cols * rows) * 3; i += 3)
	{
		adjustChannel(img[i]);
		adjustChannel(img[i + 1]);
		adjustChannel(img[i + 2]);
	}
}

void adjustBrightnessGPUOpenCL(unsigned char* img, size_t rows, size_t cols, unsigned char a, unsigned char b) 
{
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Platform platform = platforms[0];

	vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	cl::Device device = devices[0];

	cl::Context context(device);
	cl::CommandQueue queue(context, device);

	const char* kernelSource = R"(
        __kernel void adjustBrightness(
            __global unsigned char* img,
            int rows, int cols,
            unsigned char a, unsigned char b) {
            int id = get_global_id(0);
            int total = rows * cols * 3;
            if (id < total) {
                img[id] = (img[id] - a) * 255 / (b - a);
            }
        }
    )";

	cl::Program program(context, kernelSource);
	program.build(devices);
	cl::Kernel kernel(program, "adjustBrightness");

	size_t dataSize = rows * cols * 3 * sizeof(unsigned char);
	cl::Buffer d_img(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize, img);

	kernel.setArg(0, d_img);
	kernel.setArg(1, (int)rows);
	kernel.setArg(2, (int)cols);
	kernel.setArg(3, a);
	kernel.setArg(4, b);

	cl::NDRange global(rows * cols * 3);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);

	queue.enqueueReadBuffer(d_img, CL_TRUE, 0, dataSize, img);
}