#include "stereotgv.h"

__global__ void TgvCloneKernel(float* dst, float* src, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		dst[pos] = src[pos];
	}
}

void StereoTgv::Clone(float* dst, int w, int h, int s, float* src) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	TgvCloneKernel << < blocks, threads >> > (dst, src, w, h, s);
}


__global__ void TgvCloneKernel2(float2* dst, float2* src, int width, int height, int stride) {
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		dst[pos] = src[pos];
	}
}

void StereoTgv::Clone(float2* dst, int w, int h, int s, float2* src) {
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	TgvCloneKernel2 << < blocks, threads >> > (dst, src, w, h, s);
}