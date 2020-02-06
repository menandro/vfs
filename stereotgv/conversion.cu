#include "stereotgv.h"

__global__ void TgvConvertKBKernel(float2* disparity,
	float focalx, float focaly, float cx, float cy,
	float d1, float d2, float d3, float d4,
	float t1, float t2, float t3,
	float3* X, float* depth,
	int width, int height, int stride)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos = ix + iy * stride;

	if (ix >= width || iy >= height) return;

	float u0 = (float)ix;
	float v0 = (float)iy;
	float xprime0 = (u0 - focalx) / cx;
	float yprime0 = (v0 - focaly) / cy;

	float u = disparity[pos].x;
	float v = disparity[pos].y;

	float u1 = u0 + u;
	float v1 = v0 + v;
	float xprime1 = (u1 - focalx) / cx;
	float yprime1 = (v1 - focaly) / cy;

	// Newton-Raphson Method Frame 0
	float ru0 = sqrtf(xprime0 * xprime0 + yprime0 * yprime0);
	float theta0 = 0.0f;
	for (int iter = 0; iter < 5; iter++) {
		float thetad0 = theta0 + d1 * powf(theta0, 3.0f) + d2 * powf(theta0, 5.0f) + d3 * powf(theta0, 7.0f)
			+ d4 * powf(theta0, 9.0f);
		float Dthetad0 = 1.0f + 3.0f * d1 * powf(theta0, 2.0f) + 5.0f * d2 * powf(theta0, 4.0f)
			+ 7.0f * d3 * powf(theta0, 6.0f) + 9.0f * d4 * powf(theta0, 8.0f);
		float f0 = ru0 - thetad0;
		float f0prime = -Dthetad0;// 2 * (ru0 - thetad0).*(-Dthetad0);
		theta0 = theta0 - f0 / f0prime;
	}
	float x0out = tanf(theta0) * xprime0 / ru0;
	float y0out = tanf(theta0) * yprime0 / ru0;

	// Newton-Raphson Method Frame 1
	float ru1 = sqrtf(xprime1 * xprime1 + yprime1 * yprime1);
	float theta1 = 0.0f;
	for (int iter = 0; iter < 5; iter++) {
		float thetad1 = theta1 + d1 * powf(theta1, 3.0f) + d2 * powf(theta1, 5.0f) + d3 * powf(theta1, 7.0f)
			+ d4 * powf(theta1, 9.0f);
		float Dthetad1 = 1.0f + 3.0f * d1 * powf(theta1, 2.0f) + 5.0f * d2 * powf(theta1, 4.0f)
			+ 7.0f * d3 * powf(theta1, 6.0f) + 9.0f * d4 * powf(theta1, 8.0f);
		float f1 = ru1 - thetad1;// % (ru1 - thetad1). ^ 2;
		float f1prime = -Dthetad1;// % 2 * (ru1 - thetad1).*(-Dthetad1);
		theta1 = theta1 - f1 / f1prime;
	}
	float x1out = tanf(theta1) * xprime1 / ru1;
	float y1out = tanf(theta1) * yprime1 / ru1;

	// Triangulation
	float Zx = (t1 - x1out * t3) / (x1out - x0out);
	float Zy = (t2 - y1out * t3) / (y1out - y0out);
	float Z = Zx;
	X[pos].x = x0out * Z;
	X[pos].y = y0out * Z;
	X[pos].z = Z;
	depth[pos] = sqrt(X[pos].x * X[pos].x + X[pos].y * X[pos].y + X[pos].z * X[pos].z);
	//depth[pos] = Z;
}

void StereoTgv::ConvertKB(float2* disparity, float focalx, float focaly, float cx, float cy,
	float d1, float d2, float d3, float d4, float t1, float t2, float t3,
	float3* X, float* depth, int w, int h, int s)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	TgvConvertKBKernel << <blocks, threads >> > (disparity,
		focalx, focaly, cx, cy,
		d1, d2, d3, d4,
		t1, t2, t3,
		X, depth,
		w, h, s);
}