#include "stereotgv.h"

/// image to warp
texture<float, cudaTextureType2D, cudaReadModeElementType> texToWarp;
texture<float2, cudaTextureType2D, cudaReadModeElementType> texTv;
texture<float, cudaTextureType2D, cudaReadModeElementType> texTvx;
texture<float, cudaTextureType2D, cudaReadModeElementType> texTvy;

__global__ void TgvWarpingKernel(int width, int height, int stride,
	const float2 *warpUV, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float x = ((float)ix + warpUV[pos].x + 0.5f) / (float)width;
	float y = ((float)iy + warpUV[pos].y + 0.5f) / (float)height;

	out[pos] = tex2D(texToWarp, x, y);
}

void StereoTgv::WarpImage(const float *src, int w, int h, int s,
	const float2 *warpUV, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texToWarp.addressMode[0] = cudaAddressModeMirror;
	texToWarp.addressMode[1] = cudaAddressModeMirror;
	texToWarp.filterMode = cudaFilterModeLinear;
	texToWarp.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texToWarp, src, w, h, s * sizeof(float));

	TgvWarpingKernel << <blocks, threads >> > (w, h, s, warpUV, out);
}

// **************************************************
// ** Find Warping vector direction (tvx2, tvy2) for Fisheye Stereo
// **************************************************

__global__ void TgvFindWarpingVectorKernel(const float2 *warpUV,
	int width, int height, int stride, float2 *tvx2)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float x = ((float)ix + warpUV[pos].x + 0.5f) / (float)width;
	float y = ((float)iy + warpUV[pos].y + 0.5f) / (float)height;

	tvx2[pos].x = tex2D(texTvx, x, y);
	tvx2[pos].x = tex2D(texTvy, x, y);
	//tv2[pos] = make_float2(x, y);
}

void StereoTgv::FindWarpingVector(const float2 *warpUV, const float *tvx, const float *tvy,
	int w, int h, int s, float2 *tv2)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texTvx.addressMode[0] = cudaAddressModeMirror;
	texTvx.addressMode[1] = cudaAddressModeMirror;
	texTvx.filterMode = cudaFilterModeLinear;
	texTvx.normalized = true;

	texTvy.addressMode[0] = cudaAddressModeMirror;
	texTvy.addressMode[1] = cudaAddressModeMirror;
	texTvy.filterMode = cudaFilterModeLinear;
	texTvy.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texTvx, tvx, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texTvy, tvy, w, h, s * sizeof(float));

	TgvFindWarpingVectorKernel << <blocks, threads >> > (warpUV, w, h, s, tv2);
}


// **************************************************
// ** Find Warping vector direction tv2<float2> for Fisheye Stereo
// **************************************************

__global__ void TgvFindWarpingVectorFloat2Kernel(const float2 *warpUV,
	int width, int height, int stride, float2 *tv2)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float x = ((float)ix + warpUV[pos].x + 0.5f) / (float)width;
	float y = ((float)iy + warpUV[pos].y + 0.5f) / (float)height;

	tv2[pos] = tex2D(texTv, x, y);
	//tv2[pos] = make_float2(x, y);
}

void StereoTgv::FindWarpingVector(const float2 *warpUV, const float2 *tv,
	int w, int h, int s, float2 *tv2)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texTv.addressMode[0] = cudaAddressModeMirror;
	texTv.addressMode[1] = cudaAddressModeMirror;
	texTv.filterMode = cudaFilterModeLinear;
	texTv.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	cudaBindTexture2D(NULL, texTv, tv, w, h, s * sizeof(float2));

	TgvFindWarpingVectorFloat2Kernel << <blocks, threads >> > (warpUV, w, h, s, tv2);
}

// **************************************************
// ** Compute Optical flow (u,v) for Fisheye Stereo
// **************************************************

__global__ void TgvComputeOpticalFlowVectorKernel(const float *u, const float2 *tv2,
	int width, int height, int stride, float2 *warpUV)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float us = u[pos];
	float2 tv2s = tv2[pos];
	warpUV[pos].x = us * tv2s.x;
	warpUV[pos].y = us * tv2s.y;
}

void StereoTgv::ComputeOpticalFlowVector(const float *u, const float2 *tv2,
	int w, int h, int s, float2 *warpUV)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvComputeOpticalFlowVectorKernel << <blocks, threads >> > (u, tv2, w, h, s, warpUV);
}


// ******************************
// MASKED VeRsIoN
// ******************************
__global__ void TgvWarpingMaskedKernel(float* mask, int width, int height, int stride,
	const float2 *warpUV, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float x = ((float)ix + warpUV[pos].x + 0.5f) / (float)width;
	float y = ((float)iy + warpUV[pos].y + 0.5f) / (float)height;

	out[pos] = tex2D(texToWarp, x, y);
}

void StereoTgv::WarpImageMasked(const float *src, float* mask, int w, int h, int s,
	const float2 *warpUV, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texToWarp.addressMode[0] = cudaAddressModeMirror;
	texToWarp.addressMode[1] = cudaAddressModeMirror;
	texToWarp.filterMode = cudaFilterModeLinear;
	texToWarp.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texToWarp, src, w, h, s * sizeof(float));

	TgvWarpingMaskedKernel << <blocks, threads >> > (mask, w, h, s, warpUV, out);
}

// **************************************************
// ** Find Warping vector direction (tvx2, tvy2) for Fisheye Stereo
// **************************************************

__global__ void TgvFindWarpingVectorMaskedKernel(const float2 *warpUV, float* mask, 
	int width, int height, int stride, float2 *tvx2)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float x = ((float)ix + warpUV[pos].x + 0.5f) / (float)width;
	float y = ((float)iy + warpUV[pos].y + 0.5f) / (float)height;

	tvx2[pos].x = tex2D(texTvx, x, y);
	tvx2[pos].x = tex2D(texTvy, x, y);
	//tv2[pos] = make_float2(x, y);
}

void StereoTgv::FindWarpingVectorMasked(const float2 *warpUV, float* mask, const float *tvx, const float *tvy,
	int w, int h, int s, float2 *tv2)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texTvx.addressMode[0] = cudaAddressModeMirror;
	texTvx.addressMode[1] = cudaAddressModeMirror;
	texTvx.filterMode = cudaFilterModeLinear;
	texTvx.normalized = true;

	texTvy.addressMode[0] = cudaAddressModeMirror;
	texTvy.addressMode[1] = cudaAddressModeMirror;
	texTvy.filterMode = cudaFilterModeLinear;
	texTvy.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, texTvx, tvx, w, h, s * sizeof(float));
	cudaBindTexture2D(0, texTvy, tvy, w, h, s * sizeof(float));

	TgvFindWarpingVectorMaskedKernel << <blocks, threads >> > (warpUV, mask, w, h, s, tv2);
}


// **************************************************
// ** Find Warping vector direction tv2<float2> for Fisheye Stereo
// **************************************************

__global__ void TgvFindWarpingVectorFloat2MaskedKernel(const float2 *warpUV, float* mask, 
	int width, int height, int stride, float2 *tv2)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float x = ((float)ix + warpUV[pos].x + 0.5f) / (float)width;
	float y = ((float)iy + warpUV[pos].y + 0.5f) / (float)height;

	tv2[pos] = tex2D(texTv, x, y);
	//tv2[pos] = make_float2(x, y);
}

void StereoTgv::FindWarpingVectorMasked(const float2 *warpUV, float* mask, const float2 *tv,
	int w, int h, int s, float2 *tv2)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	texTv.addressMode[0] = cudaAddressModeMirror;
	texTv.addressMode[1] = cudaAddressModeMirror;
	texTv.filterMode = cudaFilterModeLinear;
	texTv.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	cudaBindTexture2D(NULL, texTv, tv, w, h, s * sizeof(float2));

	TgvFindWarpingVectorFloat2MaskedKernel << <blocks, threads >> > (warpUV, mask, w, h, s, tv2);
}

// **************************************************
// ** Compute Optical flow (u,v) for Fisheye Stereo
// **************************************************

__global__ void TgvComputeOpticalFlowVectorMaskedKernel(const float *u, const float2 *tv2, float* mask,
	int width, int height, int stride, float2 *warpUV)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float us = u[pos];
	float2 tv2s = tv2[pos];
	warpUV[pos].x = us * tv2s.x;
	warpUV[pos].y = us * tv2s.y;
}

void StereoTgv::ComputeOpticalFlowVectorMasked(const float *u, const float2 *tv2, float* mask,
	int w, int h, int s, float2 *warpUV)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvComputeOpticalFlowVectorMaskedKernel << <blocks, threads >> > (u, tv2, mask, w, h, s, warpUV);
}