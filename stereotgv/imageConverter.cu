#include "stereotgv.h"

__global__
void TgvCv8uToGrayKernel(uchar *d_iCv8u, float *d_iGray, int width, int height, int stride)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((r < height) && (c < width))
	{
		int idx = c + stride * r;        // current pixel index 

		//d_iGray[idx] = 0.2126f * (float)pixel.x + 0.7152f * (float)pixel.y + 0.0722f * (float)pixel.z;
		d_iGray[idx] = (float)d_iCv8u[idx] / 256.0f;
	}
}

void StereoTgv::Cv8uToGray(uchar * d_iCv8u, float *d_iGray, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvCv8uToGrayKernel << < blocks, threads >> > (d_iCv8u, d_iGray, w, h, s);
}

__global__
void TgvCv8uc3ToGrayKernel(uchar3 *d_iRgb, float *d_iGray, int width, int height, int stride)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((r < height) && (c < width))
	{
		int idx = c + stride * r;        // current pixel index 

		uchar3 pixel = d_iRgb[idx];

		//d_iGray[idx] = 0.2126f * (float)pixel.x + 0.7152f * (float)pixel.y + 0.0722f * (float)pixel.z;
		d_iGray[idx] = ((float)pixel.x + (float)pixel.y + (float)pixel.z) / 3;
		d_iGray[idx] = d_iGray[idx] / 256.0f;
	}
}

void StereoTgv::Cv8uc3ToGray(uchar3 * d_iRgb, float *d_iGray, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvCv8uc3ToGrayKernel << < blocks, threads >> > (d_iRgb, d_iGray, w, h, s);
}