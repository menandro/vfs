#include "stereotgv.h"

texture<float, cudaTextureType2D, cudaReadModeElementType> gray_img;
texture<float, cudaTextureType2D, cudaReadModeElementType> imgToFilter;

// Calculate anisotropic diffusion tensor
__global__ 
void TgvCalcTensorKernel(float* gray, float beta, float gamma, int size_grad,
	float* atensor, float* btensor, float* ctensor,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		float dx = 1.0f / (float)width;
		float dy = 1.0f / (float)height;

		float x = ((float)ix + 0.5f) * dx;
		float y = ((float)iy + 0.5f) * dy;

		float2 grad;
		float t0;
		// x derivative
		t0 = tex2D(gray_img, x + 1.0f * dx, y);
		t0 -= tex2D(gray_img, x, y);
		t0 = tex2D(gray_img, x + 1.0f * dx, y + 1.0f * dy);
		t0 -= tex2D(gray_img, x, y + 1.0f * dy);
		grad.x = t0;

		// y derivative
		t0 = tex2D(gray_img, x, y + 1.0f * dy);
		t0 -= tex2D(gray_img, x, y);
		t0 = tex2D(gray_img, x + 1.0f * dx, y + 1.0f * dy);
		t0 -= tex2D(gray_img, x + 1.0f * dx, y);
		grad.y = t0;

		float min_n_length = 1e-8f;
		float min_tensor_val = 1e-8f;

		float abs_img = sqrtf(grad.x*grad.x + grad.y*grad.y);
		float norm_n = abs_img;

		float2 n_normed;
		n_normed.x = grad.x / norm_n;
		n_normed.y = grad.y / norm_n;

		if (norm_n < min_n_length) {
			n_normed.x = 1.0f;
			n_normed.y = 0.0f;
		}

		float2 nT_normed;
		nT_normed.x = n_normed.y;
		nT_normed.y = -n_normed.x;

		float wtensor;
		if (expf(-beta * powf(abs_img, gamma)) > min_tensor_val) {
			wtensor = expf(-beta * powf(abs_img, gamma));
		}
		else wtensor = min_tensor_val;

		float a = wtensor * n_normed.x * n_normed.x + nT_normed.x * nT_normed.x;
		float c = wtensor * n_normed.x * n_normed.y + nT_normed.x * nT_normed.y;
		float b = wtensor * n_normed.y * n_normed.y + nT_normed.y * nT_normed.y;
		atensor[pos] = a;
		btensor[pos] = b;
		ctensor[pos] = c;
	}
}


void StereoTgv::CalcTensor(float* gray, float beta, float gamma, int size_grad,
	int w, int h, int s,
	float* a, float* b, float* c)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(h, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	gray_img.addressMode[0] = cudaAddressModeMirror;
	gray_img.addressMode[1] = cudaAddressModeMirror;
	gray_img.filterMode = cudaFilterModeLinear;
	gray_img.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, gray_img, gray, w, h, s * sizeof(float));

	TgvCalcTensorKernel << < blocks, threads >> > (gray, beta, gamma, size_grad,
		a, b, c, w, h, s);
}


// Calculate anisotropic diffusion tensor
__global__ void TgvGaussianKernel(float* input, float* output,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy < height) && (ix < width))
	{
		int pos = ix + iy * stride;

		float dx = 1.0f / (float)width;
		float dy = 1.0f / (float)height;

		float x = ((float)ix + 0.5f) * dx;
		float y = ((float)iy + 0.5f) * dy;

		float2 grad;
		float t0 = (1 / 4.0f)*tex2D(imgToFilter, x, y);
		t0 += (1 / 16.0f)*tex2D(imgToFilter, x - 1.0f * dx, y - 1.0f * dy);
		t0 += (1 / 16.0f)*tex2D(imgToFilter, x - 1.0f * dx, y + 1.0f * dy);
		t0 += (1 / 16.0f)*tex2D(imgToFilter, x + 1.0f * dx, y - 1.0f * dy);
		t0 += (1 / 16.0f)*tex2D(imgToFilter, x + 1.0f * dx, y + 1.0f * dy);
		t0 += (1 / 8.0f)*tex2D(imgToFilter, x - 1.0f * dx, y);
		t0 += (1 / 8.0f)*tex2D(imgToFilter, x + 1.0f * dx, y);
		t0 += (1 / 8.0f)*tex2D(imgToFilter, x, y - 1.0f * dy);
		t0 += (1 / 8.0f)*tex2D(imgToFilter, x, y + 1.0f * dy);

		output[pos] = t0;
	}
}


void StereoTgv::Gaussian(float* input, int w, int h, int s, float* output)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	imgToFilter.addressMode[0] = cudaAddressModeMirror;
	imgToFilter.addressMode[1] = cudaAddressModeMirror;
	imgToFilter.filterMode = cudaFilterModeLinear;
	imgToFilter.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, imgToFilter, input, w, h, s * sizeof(float));
	TgvGaussianKernel << < blocks, threads >> > (input, output,
		w, h, s);
}


// **************************
//  MASKED
// **************************
// Calculate anisotropic diffusion tensor
__global__
void TgvCalcTensorMaskedKernel(float* gray, float* mask, float beta, float gamma, int size_grad,
	float* atensor, float* btensor, float* ctensor,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float2 grad;
	float t0;
	// x derivative
	t0 = tex2D(gray_img, x + 1.0f * dx, y);
	t0 -= tex2D(gray_img, x, y);
	t0 = tex2D(gray_img, x + 1.0f * dx, y + 1.0f * dy);
	t0 -= tex2D(gray_img, x, y + 1.0f * dy);
	grad.x = t0;

	// y derivative
	t0 = tex2D(gray_img, x, y + 1.0f * dy);
	t0 -= tex2D(gray_img, x, y);
	t0 = tex2D(gray_img, x + 1.0f * dx, y + 1.0f * dy);
	t0 -= tex2D(gray_img, x + 1.0f * dx, y);
	grad.y = t0;

	float min_n_length = 1e-8f;
	float min_tensor_val = 1e-8f;

	float abs_img = sqrtf(grad.x*grad.x + grad.y*grad.y);
	float norm_n = abs_img;

	float2 n_normed;
	n_normed.x = grad.x / norm_n;
	n_normed.y = grad.y / norm_n;

	if (norm_n < min_n_length) {
		n_normed.x = 1.0f;
		n_normed.y = 0.0f;
	}

	float2 nT_normed;
	nT_normed.x = n_normed.y;
	nT_normed.y = -n_normed.x;

	float wtensor;
	if (expf(-beta * powf(abs_img, gamma)) > min_tensor_val) {
		wtensor = expf(-beta * powf(abs_img, gamma));
	}
	else wtensor = min_tensor_val;

	float a = wtensor * n_normed.x * n_normed.x + nT_normed.x * nT_normed.x;
	float c = wtensor * n_normed.x * n_normed.y + nT_normed.x * nT_normed.y;
	float b = wtensor * n_normed.y * n_normed.y + nT_normed.y * nT_normed.y;
	atensor[pos] = a;
	btensor[pos] = b;
	ctensor[pos] = c;
}


void StereoTgv::CalcTensorMasked(float* gray, float* mask, float beta, float gamma, int size_grad,
	int w, int h, int s,
	float* a, float* b, float* c)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	gray_img.addressMode[0] = cudaAddressModeMirror;
	gray_img.addressMode[1] = cudaAddressModeMirror;
	gray_img.filterMode = cudaFilterModeLinear;
	gray_img.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, gray_img, gray, w, h, s * sizeof(float));

	TgvCalcTensorMaskedKernel << < blocks, threads >> > (gray, mask, beta, gamma, size_grad,
		a, b, c, w, h, s);
}


// Calculate anisotropic diffusion tensor MASKED
__global__ void TgvGaussianMaskedKernel(float* input, float* mask, float* output,
	int width, int height, int stride)
{
	int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float dx = 1.0f / (float)width;
	float dy = 1.0f / (float)height;

	float x = ((float)ix + 0.5f) * dx;
	float y = ((float)iy + 0.5f) * dy;

	float2 grad;
	float t0 = (1 / 4.0f)*tex2D(imgToFilter, x, y);
	t0 += (1 / 16.0f)*tex2D(imgToFilter, x - 1.0f * dx, y - 1.0f * dy);
	t0 += (1 / 16.0f)*tex2D(imgToFilter, x - 1.0f * dx, y + 1.0f * dy);
	t0 += (1 / 16.0f)*tex2D(imgToFilter, x + 1.0f * dx, y - 1.0f * dy);
	t0 += (1 / 16.0f)*tex2D(imgToFilter, x + 1.0f * dx, y + 1.0f * dy);
	t0 += (1 / 8.0f)*tex2D(imgToFilter, x - 1.0f * dx, y);
	t0 += (1 / 8.0f)*tex2D(imgToFilter, x + 1.0f * dx, y);
	t0 += (1 / 8.0f)*tex2D(imgToFilter, x, y - 1.0f * dy);
	t0 += (1 / 8.0f)*tex2D(imgToFilter, x, y + 1.0f * dy);

	output[pos] = t0;
}


void StereoTgv::GaussianMasked(float* input, float* mask, int w, int h, int s, float* output)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	// mirror if a coordinate value is out-of-range
	imgToFilter.addressMode[0] = cudaAddressModeMirror;
	imgToFilter.addressMode[1] = cudaAddressModeMirror;
	imgToFilter.filterMode = cudaFilterModeLinear;
	imgToFilter.normalized = true;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaBindTexture2D(0, imgToFilter, input, w, h, s * sizeof(float));
	TgvGaussianMaskedKernel << < blocks, threads >> > (input, mask, output,
		w, h, s);
}