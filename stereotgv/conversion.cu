#include "stereotgv.h"

__global__ void TgvConvertDisparityToDepthKernel(float *disparity, float baseline,
	float focal, int width, int height, int stride, float *depth)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	/*float Z = baseline * focal / disparity[pos];
	float X = (ix - width / 2)*Z / focal;
	float Y = (iy - height / 2)*Z / focal;
	depth[pos] = sqrt(Z * Z + X * X + Y * Y);*/
	depth[pos] = baseline * focal / disparity[pos];
}


void StereoTgv::ConvertDisparityToDepth(float *disparity, float baseline, float focal, int w, int h, int s, float *depth)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvConvertDisparityToDepthKernel << <blocks, threads >> > (disparity, baseline, focal, w, h, s, depth);
}


//*******************
// Masked
//*******************
__global__ void TgvConvertDisparityToDepthMaskedKernel(float *disparity, float* mask, float baseline,
	float focal, int width, int height, int stride, float *depth)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	depth[pos] = baseline * focal / disparity[pos];
}


void StereoTgv::ConvertDisparityToDepthMasked(float *disparity, float* mask, float baseline, float focal, 
	int w, int h, int s, float *depth)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvConvertDisparityToDepthMaskedKernel << <blocks, threads >> > (disparity, mask, baseline, focal, w, h, s, depth);
}