#pragma once

#include "Bilateral_Definition.h"

__constant__ int VIDEO_XY_[2];
__constant__ double PI_;
__constant__ Bilateral BILATERAL_;

__CODE__ inline int GetIdx(const int x, const int y) {
	return y * VIDEO_XY_[0] + x;
}

__CODE__ inline float EuclideanDistance(const int x, const int y, const int i, const int j) {
	return sqrtf(pow((float)(x - i), 2.f) + pow((float)(y - j), 2.f));
}

__CODE__ inline float Gaussian(float x, float sigma) {
	return expf(-(powf(x, 2.f)) / (2 * powf(sigma, 2.f))) / (2 * PI_ * powf(sigma, 2.f));
}

__CODE__ inline int BGRToGrayscale_Device(const uchar* Input, const int idx)
{
	int GrayPixel = (int)(ceilf((int)Input[idx * 3] * .1140f + (int)Input[idx * 3 + 1] * 0.5870f +
		(int)Input[idx * 3 + 2] * 0.2989f));
	return GrayPixel;
}

__CODE__ inline int BilateralFilter_Device(const int* Input, const int idx) {
	const int x = idx % VIDEO_XY_[0], y = idx / VIDEO_XY_[0];
	const int f = BILATERAL_.Diameter / 2;

	const int Left = max(0, x - f);
	const int Right = min(x + f, VIDEO_XY_[0] - 1);
	const int Up = max(0, y - f);
	const int Down = min(y + f, VIDEO_XY_[1] - 1);

	float PixResult = 0.f;
	float WeightSum = 0.f;

	for (int i = Left; i <= Right; i++)
		for (int j = Up; j <= Down; j++) {
			const int Neighbor_Idx = GetIdx(i, j);

			float Weight = Gaussian(Input[Neighbor_Idx] - Input[idx], BILATERAL_.SigmaD) *
				Gaussian(EuclideanDistance(x, y, i, j), BILATERAL_.SigmaR);

			PixResult += Input[Neighbor_Idx] * Weight;
			WeightSum += Weight;
		}

	PixResult /= WeightSum;
	return (int)PixResult;
}