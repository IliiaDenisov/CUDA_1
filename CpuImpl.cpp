#include "Functions.h"

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