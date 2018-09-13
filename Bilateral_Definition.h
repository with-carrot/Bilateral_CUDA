#pragma once

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define __CODE__ __device__
#else
#define __CODE__
#endif

// CUDA Definitions
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// OpenCV Definitions
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

const int NUM_THREADS = 32;
const double PI = 3.14159265358979;

const dim3 ThreadsPerBlock = dim3(NUM_THREADS, NUM_THREADS);

// Bilateral Filter Struct
struct Bilateral {
	const int Diameter = 5;
	const float SigmaD = 3.f;
	const float SigmaR = 80.f;
};

const Bilateral BILATERAL;