#include "Bilateral.h"

__global__ void BGRToGrayscale(const uchar* Input, int* Output);
__global__ void BilateralFilter(const int* Input, uchar* Output);

Bilateral_Filter::Bilateral_Filter() {
	cudaMemcpyToSymbolAsync(PI_, &PI, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbolAsync(BILATERAL_, &BILATERAL, sizeof(Bilateral), 0, cudaMemcpyHostToDevice);
}

Bilateral_Filter::~Bilateral_Filter() {
	Image_Bilateral.~Mat();

	cudaFree(Image_CUDA);
	cudaFree(Gray_CUDA);
}

void Bilateral_Filter::InitSetting(const cv::Mat Image) {
	const int x = Image.cols;
	const int y = Image.rows;

	VIDEO_XY[0] = x;
	VIDEO_XY[1] = y;
	VIDEO_SIZE = x * y;

	BlocksPerGrid = dim3((int)ceilf((float)VIDEO_XY[0] / (float)ThreadsPerBlock.x),
						(int)ceilf((float)VIDEO_XY[1] / (float)ThreadsPerBlock.y));

	cudaMemcpyToSymbolAsync(VIDEO_XY_, VIDEO_XY, 2 * sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Image_CUDA, VIDEO_SIZE * 3 * sizeof(uchar));
	cudaMalloc((void**)&Gray_CUDA, VIDEO_SIZE * sizeof(int));

	Image_Bilateral = cv::Mat(cv::Size(x, y), CV_8UC1);
}

void Bilateral_Filter::Operate(const cv::Mat Image) {
	cudaMemcpyAsync(Image_CUDA, Image.data, VIDEO_SIZE * 3 * sizeof(uchar), cudaMemcpyHostToDevice);

	BGRToGrayscale_(Image_CUDA, Gray_CUDA);
	BilateralFilter_(Gray_CUDA, Image_Bilateral.data);

	cudaDeviceSynchronize();

	cv::imshow("Result", Image_Bilateral);
}

void Bilateral_Filter::BGRToGrayscale_(const uchar* CUDA_In, int* CUDA_Out) {
	BGRToGrayscale << <BlocksPerGrid, ThreadsPerBlock >> > (CUDA_In, CUDA_Out);
}

void Bilateral_Filter::BilateralFilter_(const int* CUDA_In, uchar* CUDA_Out) {
	BilateralFilter << <BlocksPerGrid, ThreadsPerBlock >> > (CUDA_In, CUDA_Out);

}

__global__ void BGRToGrayscale(const uchar* Input, int* Output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= VIDEO_XY_[0] || y >= VIDEO_XY_[1])
		return;

	int idx = y * VIDEO_XY_[0] + x;

	Output[idx] = BGRToGrayscale_Device(Input, idx);
}

__global__ void BilateralFilter(const int* Input, uchar* Output) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= VIDEO_XY_[0] || y >= VIDEO_XY_[1])
		return;

	int idx = y * VIDEO_XY_[0] + x;

	__shared__ uchar Pixel;

	if (x < 2 || x > VIDEO_XY_[0] - 3 || y < 2 || y > VIDEO_XY_[1] - 3) {
		Pixel = (uchar)Input[idx];
		Output[idx] = Pixel;
		return;
	}
	
	Pixel = (uchar)BilateralFilter_Device(Input, idx);
	Output[idx] = Pixel;
}