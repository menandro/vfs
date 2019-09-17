#include "stereotgv.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> texI0;
texture<float, cudaTextureType2D, cudaReadModeElementType> texI1;

__global__ void TgvComputeDerivativesKernel(int width, int height, int stride,
	float *Ix, float *Iy, float *Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float t0, t1;
	// x derivative
	t0 = tex2D(texI0, x - 2.0f * dx, y);
	t0 -= tex2D(texI0, x - 1.0f * dx, y) * 8.0f;
	t0 += tex2D(texI0, x + 1.0f * dx, y) * 8.0f;
	t0 -= tex2D(texI0, x + 2.0f * dx, y);
	t0 /= 12.0f;

	t1 = tex2D(texI1, x - 2.0f * dx, y);
	t1 -= tex2D(texI1, x - 1.0f * dx, y) * 8.0f;
	t1 += tex2D(texI1, x + 1.0f * dx, y) * 8.0f;
	t1 -= tex2D(texI1, x + 2.0f * dx, y);
	t1 /= 12.0f;

	Ix[pos] = (t0 + t1) * 0.5f;

	// t derivative
	Iz[pos] = tex2D(texI1, x, y) - tex2D(texI0, x, y);

	// y derivative
	t0 = tex2D(texI0, x, y - 2.0f * dy);
	t0 -= tex2D(texI0, x, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texI0, x, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texI0, x, y + 2.0f * dy);
	t0 /= 12.0f;

	t1 = tex2D(texI1, x, y - 2.0f * dy);
	t1 -= tex2D(texI1, x, y - 1.0f * dy) * 8.0f;
	t1 += tex2D(texI1, x, y + 1.0f * dy) * 8.0f;
	t1 -= tex2D(texI1, x, y + 2.0f * dy);
	t1 /= 12.0f;

	Iy[pos] = (t0 + t1) * 0.5f;
}

///CUDA CALL FUNCTIONS ***********************************************************
void StereoTgv::ComputeDerivatives(float *I0, float *I1,
	int w, int h, int s,
	float *Ix, float *Iy, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texI0.addressMode[0] = cudaAddressModeMirror;
	texI0.addressMode[1] = cudaAddressModeMirror;
	texI0.filterMode = cudaFilterModeLinear;
	texI0.normalized = true;

	texI1.addressMode[0] = cudaAddressModeMirror;
	texI1.addressMode[1] = cudaAddressModeMirror;
	texI1.filterMode = cudaFilterModeLinear;
	texI1.normalized = true;

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texI0, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texI1, I1, w, h, s * sizeof(float));

	TgvComputeDerivativesKernel << < blocks, threads >> > (w, h, s, Ix, Iy, Iz);
}


//****************************************
// Fisheye Stereo 1D Derivative
//****************************************
__global__
void TgvComputeDerivativesFisheyeKernel(float2 * vector, int width, int height, int stride,
	float *Iw, float *Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float vx = vector[pos].x;
	float vy = vector[pos].y;
	float r = sqrtf(vx * vx + vy * vy);

	// Normalize because pyramid sampling ruins normality
	float dx = (vx / r) / (float)width;
	float dy = (vy / r) / (float)height;

	float x = ((float)ix + 0.5f) / (float)width;
	float y = ((float)iy + 0.5f) / (float)height;

	float t0;
	// curve w derivative
	t0 = tex2D(texI0, x - 2.0f * dx, y - 2.0f * dy);
	t0 -= tex2D(texI0, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texI0, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texI0, x + 2.0f * dx, y + 2.0f * dy);
	t0 /= 12.0f;

	float t1;
	t1 = tex2D(texI1, x - 2.0f * dx, y - 2.0f * dy);
	t1 -= tex2D(texI1, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t1 += tex2D(texI1, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t1 -= tex2D(texI1, x + 2.0f * dx, y + 2.0f * dy);
	t1 /= 12.0f;

	Iw[pos] = (t0 + t1) * 0.5f;

	// t derivative
	Iz[pos] = tex2D(texI1, x, y) - tex2D(texI0, x, y);
}

///CUDA CALL FUNCTIONS ***********************************************************
void StereoTgv::ComputeDerivativesFisheye(float *I0, float *I1, float2 *vector,
	int w, int h, int s, float *Iw, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texI0.addressMode[0] = cudaAddressModeMirror;
	texI0.addressMode[1] = cudaAddressModeMirror;
	texI0.filterMode = cudaFilterModeLinear;
	texI0.normalized = true;

	texI1.addressMode[0] = cudaAddressModeMirror;
	texI1.addressMode[1] = cudaAddressModeMirror;
	texI1.filterMode = cudaFilterModeLinear;
	texI1.normalized = true;

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texI0, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texI1, I1, w, h, s * sizeof(float));

	TgvComputeDerivativesFisheyeKernel << < blocks, threads >> > (vector, w, h, s, Iw, Iz);
}


//****************************************
// Fisheye Stereo 1D Derivative MASKED
//****************************************
__global__
void TgvComputeDerivativesFisheyeMaskedKernel(float2 * vector, float* mask, int width, int height, int stride,
	float *Iw, float *Iz)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;
	
	float vx = vector[pos].x;
	float vy = vector[pos].y;
	float r = sqrtf(vx * vx + vy * vy);

	// Normalize because pyramid sampling ruins normality
	float dx = (vx / r) / (float)width;
	float dy = (vy / r) / (float)height;

	float x = ((float)ix + 0.5f) / (float)width;
	float y = ((float)iy + 0.5f) / (float)height;

	float t0;
	// curve w derivative
	t0 = tex2D(texI0, x - 2.0f * dx, y - 2.0f * dy);
	t0 -= tex2D(texI0, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t0 += tex2D(texI0, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t0 -= tex2D(texI0, x + 2.0f * dx, y + 2.0f * dy);
	t0 /= 12.0f;

	float t1;
	t1 = tex2D(texI1, x - 2.0f * dx, y - 2.0f * dy);
	t1 -= tex2D(texI1, x - 1.0f * dx, y - 1.0f * dy) * 8.0f;
	t1 += tex2D(texI1, x + 1.0f * dx, y + 1.0f * dy) * 8.0f;
	t1 -= tex2D(texI1, x + 2.0f * dx, y + 2.0f * dy);
	t1 /= 12.0f;

	Iw[pos] = (t0 + t1) * 0.5f;

	// t derivative
	Iz[pos] = tex2D(texI1, x, y) - tex2D(texI0, x, y);
}

///CUDA CALL FUNCTIONS ***********************************************************
void StereoTgv::ComputeDerivativesFisheyeMasked(float *I0, float *I1, float2 *vector, float* mask,
	int w, int h, int s, float *Iw, float *Iz)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texI0.addressMode[0] = cudaAddressModeMirror;
	texI0.addressMode[1] = cudaAddressModeMirror;
	texI0.filterMode = cudaFilterModeLinear;
	texI0.normalized = true;

	texI1.addressMode[0] = cudaAddressModeMirror;
	texI1.addressMode[1] = cudaAddressModeMirror;
	texI1.filterMode = cudaFilterModeLinear;
	texI1.normalized = true;

	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texI0, I0, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texI1, I1, w, h, s * sizeof(float));

	TgvComputeDerivativesFisheyeMaskedKernel << < blocks, threads >> > (vector, mask, w, h, s, Iw, Iz);
}
