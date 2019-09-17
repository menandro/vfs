#include "stereotgv.h"

__global__ void TgvSolveEtaKernel(float alpha0, float alpha1,
	float* atensor, float *btensor, float* ctensor,
	float* etau, float* etav1, float* etav2,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;
		float a = atensor[pos];
		float b = btensor[pos];
		float c = ctensor[pos];

		etau[pos] = (a*a + b * b + 2 * c*c + (a + c)*(a + c) + (b + c)*(b + c)) * (alpha1 * alpha1);
		etav1[pos] = (alpha1 * alpha1)*(b * b + c * c) + 4 * alpha0 * alpha0;
		etav2[pos] = (alpha1 * alpha1)*(a * a + c * c) + 4 * alpha0 * alpha0;
	}
}

void StereoTgv::SolveEta(float alpha0, float alpha1,
	float* a, float *b, float* c,
	int w, int h, int s,
	float* etau, float* etav1, float* etav2)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	TgvSolveEtaKernel << < blocks, threads >> > (alpha0, alpha1,
		a, b, c,
		etau, etav1, etav2,
		w, h, s);
}


// *****************************
// Masked
// *****************************
__global__ void TgvSolveEtaMaskedKernel(float* mask, float alpha0, float alpha1,
	float* atensor, float *btensor, float* ctensor,
	float* etau, float* etav1, float* etav2,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float a = atensor[pos];
	float b = btensor[pos];
	float c = ctensor[pos];

	etau[pos] = (a*a + b * b + 2 * c*c + (a + c)*(a + c) + (b + c)*(b + c)) * (alpha1 * alpha1);
	etav1[pos] = (alpha1 * alpha1)*(b * b + c * c) + 4 * alpha0 * alpha0;
	etav2[pos] = (alpha1 * alpha1)*(a * a + c * c) + 4 * alpha0 * alpha0;
}

void StereoTgv::SolveEtaMasked(float* mask, float alpha0, float alpha1,
	float* a, float *b, float* c,
	int w, int h, int s,
	float* etau, float* etav1, float* etav2)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));
	TgvSolveEtaMaskedKernel << < blocks, threads >> > (mask, alpha0, alpha1,
		a, b, c,
		etau, etav1, etav2,
		w, h, s);
}