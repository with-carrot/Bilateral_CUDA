#include <chrono>

#include "Bilateral.h"

int main() {
	Bilateral_Filter* B = new Bilateral_Filter();
	cv::VideoCapture* V = new cv::VideoCapture(3);
	cv::Mat Image;

	V->read(Image);
	B->InitSetting(Image);

	while (true) {
		V->read(Image);

		std::chrono::system_clock::time_point Start = std::chrono::system_clock::now();

		B->Operate(Image);

		std::chrono::system_clock::time_point End = std::chrono::system_clock::now();
		std::chrono::microseconds micro = std::chrono::duration_cast<std::chrono::microseconds>(End - Start);

		std::cout << "Time: " << micro.count() / 1000.0 << "ms." << std::endl;

		if (cv::waitKey(1) >= 0)
			break;
	}

	cv::destroyAllWindows();
	B->~Bilateral_Filter();

	Image.~Mat();

	free(V);
	free(B);

	return 0;
}