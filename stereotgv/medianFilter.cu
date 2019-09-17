#include "stereotgv.h"

__global__
void TgvMedianFilterKernel5(float* u, float* v,
	int width, int height, int stride,
	float *outputu, float *outputv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float mu[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	float mv[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			//get values
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[col + stride * row];
				mv[j * 5 + i] = v[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[stride*row];
				mv[j * 5 + i] = v[stride*row];
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[width - 1 + stride * row];
				mv[j * 5 + i] = v[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mu[j * 5 + i] = u[col];
				mv[j * 5 + i] = v[col];
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				mu[j * 5 + i] = u[col + stride * (height - 1)];
				mv[j * 5 + i] = v[col + stride * (height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpu, tmpv;
	for (int j = 0; j < 13; j++) {
		for (int i = j + 1; i < 25; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
			}
			if (mv[j] > mv[i]) {
				//Swap the variables.
				tmpv = mv[j];
				mv[j] = mv[i];
				mv[i] = tmpv;
			}
		}
	}

	outputu[pos] = mu[12];
	outputv[pos] = mv[12];
}

__global__ void TgvMedianFilterKernel3(float* u, float* v,
	int width, int height, int stride,
	float *outputu, float *outputv)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float mu[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	float mv[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			//get values
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			int index = j * 3 + i;
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mu[index] = u[col + stride * row];
				mv[index] = v[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mu[index] = u[stride*row];
				mv[index] = v[stride*row];
			}
			else if ((col > width) && (row >= 0) && (row < height)) {
				mu[index] = u[width - 1 + stride * row];
				mv[index] = v[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mu[index] = u[col];
				mv[index] = v[col];
			}
			else if ((col >= 0) && (col < width) && (row > height)) {
				mu[index] = u[col + stride * (height - 1)];
				mv[index] = v[col + stride * (height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpu, tmpv;
	for (int j = 0; j < 9; j++) {
		for (int i = j + 1; i < 9; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
			}
			if (mv[j] > mv[i]) {
				//Swap the variables.
				tmpv = mv[j];
				mv[j] = mv[i];
				mv[i] = tmpv;
			}
		}
	}

	outputu[pos] = mu[4];
	outputv[pos] = mv[4];
}


void StereoTgv::MedianFilter(float *inputu, float *inputv,
	int w, int h, int s,
	float *outputu, float*outputv,
	int kernelsize)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	if (kernelsize == 3) {
		TgvMedianFilterKernel3 << < blocks, threads >> > (inputu, inputv,
			w, h, s, outputu, outputv);
	}
	else if (kernelsize == 5) {
		TgvMedianFilterKernel5 << < blocks, threads >> > (inputu, inputv,
			w, h, s, outputu, outputv);
	}
}

//*************************************
//Median filter for XYZ
//*************************************
__global__ void TgvMedianFilter3DKernel3(float* X, float* Y, float *Z,
	int width, int height, int stride,
	float *X1, float *Y1, float *Z1)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float mX[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float mY[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float mZ[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			//get values
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			int index = j * 3 + i;
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mX[index] = X[col + stride * row];
				mY[index] = Y[col + stride * row];
				mZ[index] = Z[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mX[index] = X[stride*row];
				mY[index] = Y[stride*row];
				mZ[index] = Z[stride*row];
			}
			else if ((col > width) && (row >= 0) && (row < height)) {
				mX[index] = X[width - 1 + stride * row];
				mY[index] = Y[width - 1 + stride * row];
				mZ[index] = Z[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mX[index] = X[col];
				mY[index] = Y[col];
				mZ[index] = Z[col];
			}
			else if ((col >= 0) && (col < width) && (row > height)) {
				mX[index] = X[col + stride * (height - 1)];
				mY[index] = Y[col + stride * (height - 1)];
				mZ[index] = Z[col + stride * (height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpX, tmpY, tmpZ;
	for (int j = 0; j < 5; j++) {
		for (int i = j + 1; i < 9; i++) {
			if (mX[j] > mX[i]) {
				//Swap the variables.
				tmpX = mX[j];
				mX[j] = mX[i];
				mX[i] = tmpX;
			}
			if (mY[j] > mY[i]) {
				//Swap the variables.
				tmpY = mY[j];
				mY[j] = mY[i];
				mY[i] = tmpY;
			}
			if (mZ[j] > mZ[i]) {
				//Swap the variables.
				tmpZ = mZ[j];
				mZ[j] = mZ[i];
				mZ[i] = tmpZ;
			}
		}
	}

	X1[pos] = mX[4];
	Y1[pos] = mY[4];
	Z1[pos] = mZ[4];
}


void StereoTgv::MedianFilter3D(float *X, float *Y, float *Z,
	int w, int h, int s,
	float *X1, float *Y1, float *Z1,
	int kernelsize)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	if (kernelsize == 3) {
		TgvMedianFilter3DKernel3 << < blocks, threads >> > (X, Y, Z,
			w, h, s, X1, Y1, Z1);
	}
}

//*************************************
//Median filter for 1D disparity
//*************************************
__global__
void TgvMedianFilterDisparityKernel5(float* u,
	int width, int height, int stride,
	float *outputu)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float mu[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			//get values
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[stride*row];
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mu[j * 5 + i] = u[col];
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				mu[j * 5 + i] = u[col + stride * (height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpu, tmpv;
	for (int j = 0; j < 13; j++) {
		for (int i = j + 1; i < 25; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
			}
		}
	}

	outputu[pos] = mu[12];
}

__global__ void TgvMedianFilterDisparityKernel3(float* u,
	int width, int height, int stride,
	float *outputu)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float mu[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			//get values
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			int index = j * 3 + i;
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mu[index] = u[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mu[index] = u[stride*row];
			}
			else if ((col > width) && (row >= 0) && (row < height)) {
				mu[index] = u[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mu[index] = u[col];
			}
			else if ((col >= 0) && (col < width) && (row > height)) {
				mu[index] = u[col + stride * (height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpu, tmpv;
	for (int j = 0; j < 9; j++) {
		for (int i = j + 1; i < 9; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
			}
		}
	}

	outputu[pos] = mu[4];
}


void StereoTgv::MedianFilterDisparity(float *inputu,
	int w, int h, int s,
	float *outputu,
	int kernelsize)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	if (kernelsize == 3) {
		TgvMedianFilterDisparityKernel3 << < blocks, threads >> > (inputu,
			w, h, s, outputu);
	}
	else if (kernelsize == 5) {
		TgvMedianFilterDisparityKernel5 << < blocks, threads >> > (inputu,
			w, h, s, outputu);
	}
}


//*************************************
//Median filter for 1D disparity Masked
//*************************************
__global__
void TgvMedianFilterDisparityMaskedKernel5(float* u, float* mask,
	int width, int height, int stride,
	float *outputu)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float mu[25] = { 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0 };

	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++) {
			//get values
			int col = (ix + i - 2);
			int row = (iy + j - 2);
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[stride*row];
			}
			else if ((col >= width) && (row >= 0) && (row < height)) {
				mu[j * 5 + i] = u[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mu[j * 5 + i] = u[col];
			}
			else if ((col >= 0) && (col < width) && (row >= height)) {
				mu[j * 5 + i] = u[col + stride * (height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpu, tmpv;
	for (int j = 0; j < 13; j++) {
		for (int i = j + 1; i < 25; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
			}
		}
	}

	outputu[pos] = mu[12];
}

__global__ void TgvMedianFilterDisparityMaskedKernel3(float* u, float* mask,
	int width, int height, int stride,
	float *outputu)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((iy >= height) && (ix >= width)) return;
	int pos = ix + iy * stride;
	if (mask[pos] == 0.0f) return;

	float mu[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 3; i++) {
			//get values
			int col = (ix + i - 1);
			int row = (iy + j - 1);
			int index = j * 3 + i;
			if ((col >= 0) && (col < width) && (row >= 0) && (row < height)) {
				mu[index] = u[col + stride * row];
			}
			else if ((col < 0) && (row >= 0) && (row < height)) {
				mu[index] = u[stride*row];
			}
			else if ((col > width) && (row >= 0) && (row < height)) {
				mu[index] = u[width - 1 + stride * row];
			}
			else if ((col >= 0) && (col < width) && (row < 0)) {
				mu[index] = u[col];
			}
			else if ((col >= 0) && (col < width) && (row > height)) {
				mu[index] = u[col + stride * (height - 1)];
			}
			//solve gaussian
		}
	}

	float tmpu, tmpv;
	for (int j = 0; j < 9; j++) {
		for (int i = j + 1; i < 9; i++) {
			if (mu[j] > mu[i]) {
				//Swap the variables.
				tmpu = mu[j];
				mu[j] = mu[i];
				mu[i] = tmpu;
			}
		}
	}

	outputu[pos] = mu[4];
}


void StereoTgv::MedianFilterDisparityMasked(float *inputu, float* mask, 
	int w, int h, int s,
	float *outputu,
	int kernelsize)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	if (kernelsize == 3) {
		TgvMedianFilterDisparityMaskedKernel3 << < blocks, threads >> > (inputu, mask,
			w, h, s, outputu);
	}
	else if (kernelsize == 5) {
		TgvMedianFilterDisparityMaskedKernel5 << < blocks, threads >> > (inputu, mask,
			w, h, s, outputu);
	}
}