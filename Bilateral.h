#pragma once

#include "Bilateral_Function.h"

class Bilateral_Filter {
public:
	Bilateral_Filter();
	~Bilateral_Filter();

	void InitSetting(const cv::Mat Image);
	void Operate(const cv::Mat Image);

private:
	void BGRToGrayscale_(const uchar* CUDA_In, int* CUDA_Out);
	void BilateralFilter_(const int* CUDA_In, uchar* CUDA_Out);
	
	dim3 BlocksPerGrid;
	int VIDEO_XY[2];
	int VIDEO_SIZE;

	uchar* Image_CUDA = 0;
	int* Gray_CUDA = 0;
	cv::Mat Image_Bilateral;
};