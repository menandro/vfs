#include "stereotgv.h"

/// image to downscale
texture<float, cudaTextureType2D, cudaReadModeElementType> texFine;
texture<float2, cudaTextureType2D, cudaReadModeElementType> texFineFloat2;

// *********************************
// Downscaling
// *********************************
__global__ void TgvDownscaleKernel(int width, int height, int stride, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	int pos = ix + iy * stride;

	out[pos] = 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
		tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));
}

void StereoTgv::Downscale(const float *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFine.addressMode[0] = cudaAddressModeMirror;
	texFine.addressMode[1] = cudaAddressModeMirror;
	texFine.filterMode = cudaFilterModeLinear;
	texFine.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

	TgvDownscaleKernel << < blocks, threads >> > (newWidth, newHeight, newStride, out);
}


__global__ void TgvDownscaleNearestNeighborKernel(int width, int height, int stride, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	int pos = ix + iy * stride;

	out[pos] = tex2D(texFine, x, y);
	/*out[pos] = 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
		tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));*/
}

void StereoTgv::DownscaleNearestNeighbor(const float *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFine.addressMode[0] = cudaAddressModeMirror;
	texFine.addressMode[1] = cudaAddressModeMirror;
	texFine.filterMode = cudaFilterModePoint;
	texFine.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

	TgvDownscaleNearestNeighborKernel << < blocks, threads >> > (newWidth, newHeight, newStride, out);
}


// *********************************
// Downscaling for Float2
// *********************************
__global__ void TgvDownscaleKernel(int width, int height, int stride, float2 *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	int pos = ix + iy * stride;

	float2 val00 = tex2D(texFineFloat2, x - dx * 0.25f, y);
	float2 val01 = tex2D(texFineFloat2, x + dx * 0.25f, y);
	float2 val10 = tex2D(texFineFloat2, x, y - dy * 0.25f);
	float2 val11 = tex2D(texFineFloat2, x, y + dy * 0.25f);
	out[pos].x = 0.25f * (val00.x + val01.x + val10.x + val11.x);
	out[pos].y = 0.25f * (val00.y + val01.y + val10.y + val11.y);
}

void StereoTgv::Downscale(const float2 *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float2 *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFineFloat2.addressMode[0] = cudaAddressModeMirror;
	texFineFloat2.addressMode[1] = cudaAddressModeMirror;
	texFineFloat2.filterMode = cudaFilterModeLinear;
	texFineFloat2.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	checkCudaErrors(cudaBindTexture2D(0, texFineFloat2, src, width, height, stride * sizeof(float2)));

	TgvDownscaleKernel << < blocks, threads >> > (newWidth, newHeight, newStride, out);
}


// ***********************************
// Downscale with vector downscaling
//************************************

__global__ void TgvDownscaleScalingKernel(int width, int height, int stride, float scale, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	int pos = ix + iy * stride;

	out[pos] = scale * 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
		tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));
}

void StereoTgv::Downscale(const float *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float scale, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFine.addressMode[0] = cudaAddressModeMirror;
	texFine.addressMode[1] = cudaAddressModeMirror;
	texFine.filterMode = cudaFilterModeLinear;
	texFine.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

	TgvDownscaleScalingKernel << < blocks, threads >> > (newWidth, newHeight, newStride, scale, out);
}


// ***********************************
// Downscale with vector downscaling for Float2
//************************************

__global__ void TgvDownscaleScalingKernel(int width, int height, int stride, float scale, float2 *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if (ix >= width || iy >= height)
	{
		return;
	}

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	int pos = ix + iy * stride;

	float2 val00 = tex2D(texFineFloat2, x - dx * 0.25f, y);
	float2 val01 = tex2D(texFineFloat2, x + dx * 0.25f, y);
	float2 val10 = tex2D(texFineFloat2, x, y - dy * 0.25f);
	float2 val11 = tex2D(texFineFloat2, x, y + dy * 0.25f);
	out[pos].x = scale * 0.25f * (val00.x + val01.x + val10.x + val11.x);
	out[pos].y = scale * 0.25f * (val00.y + val01.y + val10.y + val11.y);
}

void StereoTgv::Downscale(const float2 *src, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float scale, float2 *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFineFloat2.addressMode[0] = cudaAddressModeMirror;
	texFineFloat2.addressMode[1] = cudaAddressModeMirror;
	texFineFloat2.filterMode = cudaFilterModeLinear;
	texFineFloat2.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	checkCudaErrors(cudaBindTexture2D(0, texFineFloat2, src, width, height, stride * sizeof(float2)));

	TgvDownscaleScalingKernel << < blocks, threads >> > (newWidth, newHeight, newStride, scale, out);
}



// ************************************
// Masked Versions
// ************************************
// *********************************
// Downscaling
// *********************************
__global__ void TgvDownscaleMaskedKernel(float* mask, int width, int height, int stride, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	out[pos] = tex2D(texFine, x, y);
	/*out[pos] = 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
		tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));*/
}

void StereoTgv::DownscaleMasked(const float *src, float *mask, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFine.addressMode[0] = cudaAddressModeMirror;
	texFine.addressMode[1] = cudaAddressModeMirror;
	texFine.filterMode = cudaFilterModeLinear;
	texFine.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

	TgvDownscaleMaskedKernel << < blocks, threads >> > (mask, newWidth, newHeight, newStride, out);
}


// *********************************
// Downscaling for Float2
// *********************************
__global__ void TgvDownscaleMaskedKernel(float* mask, int width, int height, int stride, float2 *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	/*float2 val00 = tex2D(texFineFloat2, x - dx * 0.25f, y);
	float2 val01 = tex2D(texFineFloat2, x + dx * 0.25f, y);
	float2 val10 = tex2D(texFineFloat2, x, y - dy * 0.25f);
	float2 val11 = tex2D(texFineFloat2, x, y + dy * 0.25f);
	out[pos].x = 0.25f * (val00.x + val01.x + val10.x + val11.x);
	out[pos].y = 0.25f * (val00.y + val01.y + val10.y + val11.y);*/
	out[pos] = tex2D(texFineFloat2, x, y);
}

void StereoTgv::DownscaleMasked(const float2 *src, float * mask, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float2 *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFineFloat2.addressMode[0] = cudaAddressModeMirror;
	texFineFloat2.addressMode[1] = cudaAddressModeMirror;
	texFineFloat2.filterMode = cudaFilterModeLinear;
	texFineFloat2.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	checkCudaErrors(cudaBindTexture2D(0, texFineFloat2, src, width, height, stride * sizeof(float2)));

	TgvDownscaleMaskedKernel << < blocks, threads >> > (mask, newWidth, newHeight, newStride, out);
}


// ***********************************
// Downscale with vector downscaling
//************************************

__global__ void TgvDownscaleScalingMaskedKernel(float* mask, int width, int height, int stride, float scale, float *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	out[pos] = scale * tex2D(texFine, x, y);
	/*out[pos] = scale * 0.25f * (tex2D(texFine, x - dx * 0.25f, y) + tex2D(texFine, x + dx * 0.25f, y) +
		tex2D(texFine, x, y - dy * 0.25f) + tex2D(texFine, x, y + dy * 0.25f));*/
}

void StereoTgv::DownscaleMasked(const float *src, float* mask, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float scale, float *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFine.addressMode[0] = cudaAddressModeMirror;
	texFine.addressMode[1] = cudaAddressModeMirror;
	texFine.filterMode = cudaFilterModeLinear;
	texFine.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	checkCudaErrors(cudaBindTexture2D(0, texFine, src, width, height, stride * sizeof(float)));

	TgvDownscaleScalingMaskedKernel << < blocks, threads >> > (mask, newWidth, newHeight, newStride, scale, out);
}


// ***********************************
// Downscale with vector downscaling for Float2
//************************************

__global__ void TgvDownscaleScalingMaskedKernel(float* mask, int width, int height, int stride, float scale, float2 *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float2 val = tex2D(texFineFloat2, x, y);
	out[pos].x = scale * val.x;
	out[pos].y = scale * val.y;
	/*float2 val00 = tex2D(texFineFloat2, x - dx * 0.25f, y);
	float2 val01 = tex2D(texFineFloat2, x + dx * 0.25f, y);
	float2 val10 = tex2D(texFineFloat2, x, y - dy * 0.25f);
	float2 val11 = tex2D(texFineFloat2, x, y + dy * 0.25f);
	out[pos].x = scale * 0.25f * (val00.x + val01.x + val10.x + val11.x);
	out[pos].y = scale * 0.25f * (val00.y + val01.y + val10.y + val11.y);*/
}

void StereoTgv::DownscaleMasked(const float2 *src, float * mask, int width, int height, int stride,
	int newWidth, int newHeight, int newStride, float scale, float2 *out)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(newWidth, threads.x), iDivUp(newHeight, threads.y));

	// mirror if a coordinate value is out-of-range
	texFineFloat2.addressMode[0] = cudaAddressModeMirror;
	texFineFloat2.addressMode[1] = cudaAddressModeMirror;
	texFineFloat2.filterMode = cudaFilterModeLinear;
	texFineFloat2.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

	checkCudaErrors(cudaBindTexture2D(0, texFineFloat2, src, width, height, stride * sizeof(float2)));

	TgvDownscaleScalingMaskedKernel << < blocks, threads >> > (mask, newWidth, newHeight, newStride, scale, out);
}