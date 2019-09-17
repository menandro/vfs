#include "stereotgv.h"

__global__
void ComputeColorKernel(float2 *uv, int width, int height, int stride, float3 *uvRGB, float flowscale) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;        // current row 
	int c = blockIdx.x * blockDim.x + threadIdx.x;        // current column 

	if ((r < height) && (c < width))
	{
		int pos = c + stride * r;
		float du = uv[pos].x / flowscale;
		float dv = uv[pos].y / flowscale;

		int ncols = 55;
		float rad = sqrtf(du * du + dv * dv);
		float a = atan2(-dv, -du) / 3.14159f;
		float fk = (a + 1) / 2 * ((float)ncols - 1) + 1;
		int k0 = floorf(fk); //colorwheel index lower bound
		int k1 = k0 + 1; //colorwheel index upper bound
		if (k1 == ncols + 1) {
			k1 = 1;
		}
		float f = fk - (float)k0;

		float colorwheelR[55] = { 255, 255,	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
			255, 213, 170, 128, 85, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 19, 39, 58, 78, 98, 117, 137, 156,
			176, 196, 215, 235, 255, 255, 255, 255, 255, 255 };
		float colorwheelG[55] = { 0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238,
			255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 232, 209, 186, 163,
			140, 116, 93, 70, 47, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		float colorwheelB[55] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 63, 127, 191, 255, 255, 255, 255, 255,
			255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
			255, 255, 255, 255, 255, 213, 170, 128, 85, 43 };
		/*float colorwheel[165] = { 255, 0, 0,
			255, 17, 0,
			255, 34, 0,
			255, 51, 0,
			255, 68, 0,
			255, 85, 0,
			255, 102, 0,
			255, 119, 0,
			255, 136, 0,
			255, 153, 0,
			255, 170, 0,
			255, 187, 0,
			255, 204, 0,
			255, 221, 0,
			255, 238, 0,
			255, 255, 0,
			213, 255, 0,
			170, 255, 0,
			128, 255, 0,
			85, 255, 0,
			43, 255, 0,
			0, 255, 0,
			0, 255, 63,
			0, 255, 127,
			0, 255, 191,
			0, 255, 255,
			0, 232, 255,
			0, 209, 255,
			0, 186, 255,
			0, 163, 255,
			0, 140, 255,
			0, 116, 255,
			0, 93, 255,
			0, 70, 255,
			0, 47, 255,
			0, 24, 255,
			0, 0, 255,
			19, 0, 255,
			39, 0, 255,
			58, 0, 255,
			78, 0, 255,
			98, 0, 255,
			117, 0, 255,
			137, 0, 255,
			156, 0, 255,
			176, 0, 255,
			196, 0, 255,
			215, 0, 255,
			235, 0, 255,
			255, 0, 255,
			255, 0, 213,
			255, 0, 170,
			255, 0, 128,
			255, 0, 85,
			255, 0, 43 };*/

		float colR = (1 - f) * (colorwheelR[k0] / 255.0f) + f * (colorwheelR[k1] / 255.0f);
		float colG = (1 - f) * (colorwheelG[k0] / 255.0f) + f * (colorwheelG[k1] / 255.0f);
		float colB = (1 - f) * (colorwheelB[k0] / 255.0f) + f * (colorwheelB[k1] / 255.0f);

		if (rad <= 1) {
			colR = 1 - rad * (1 - colR);
			colG = 1 - rad * (1 - colG);
			colB = 1 - rad * (1 - colB);
		}
		else {
			colR = colR * 0.75;
			colG = colG * 0.75;
			colB = colB * 0.75;
		}

		uvRGB[pos].z = (colR);
		uvRGB[pos].y = (colG);
		uvRGB[pos].x = (colB);
	}
}

void StereoTgv::FlowToHSV(float2* uv, int w, int h, int s, float3 * uRGB, float flowscale)
{
	dim3 threads(BlockWidth, BlockHeight);
	dim3 blocks(iDivUp(w, threads.x), iDivUp(h, threads.y));

	//flowToHSVKernel << < blocks, threads >> >(u, v, w, h, s, uRGB, flowscale);
	ComputeColorKernel << < blocks, threads >> > (uv, w, h, s, uRGB, flowscale);
}