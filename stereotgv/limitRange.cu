#include "stereotgv.h"

__global__
void TgvLimitRangeKernel(float* src, float upperLimit,
	int width, int height, int stride,
	float *dst)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	/*if (src[pos] < 0.0f) {
		dst[pos] = 0.0f;
	}*/
	if (src[pos] > upperLimit) {
		dst[pos] = upperLimit;
	}
	else {
		dst[pos] = src[pos];
	}
}

void StereoTgv::LimitRange(float *src, float upperLimit, int w, int h, int s, float *dst)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvLimitRangeKernel << <blocks, threads >> > (src, upperLimit, w, h, s, dst);
}