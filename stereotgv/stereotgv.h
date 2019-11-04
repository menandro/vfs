#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

#include "lib_link.h"

class StereoTgv {
public:
	StereoTgv();
	StereoTgv(int blockWidth, int blockHeight, int strideAlignment);
	~StereoTgv() {};

	int BlockWidth, BlockHeight, StrideAlignment;

	int width;
	int height;
	int stride;
	int dataSize8u;
	int dataSize8uc3;
	int dataSize32f;
	int dataSize32fc2;
	int dataSize32fc3;
	int dataSize32fc4;
	float baseline;
	float focal;
	bool visualizeResults;

	float beta;
	float gamma;
	float alpha0;
	float alpha1;
	float timestep_lambda;
	float lambda;
	float fScale;
	int nLevels;
	int nSolverIters;
	int nWarpIters;
	float limitRange = 1.0f;

	// Inputs and Outputs
	float *d_i0, *d_i1, *d_i1warp;
	uchar3 *d_i08uc3, *d_i18uc3;
	uchar *d_i08u, *d_i18u;

	float *d_i0smooth, *d_i1smooth;
	float *d_Iu, *d_Iz;
	// Output Disparity
	float* d_u, *d_du, *d_us;
	// Output Depth
	float* d_depth;
	cv::Mat depth;
	// Warping Variables
	float2 *d_warpUV, *d_warpUVs, *d_dwarpUV;
	cv::Mat warpUV, warpUVrgb;
	float3 *d_uvrgb;
	//float *d_warpX, *d_warpXs, *d_dwarpX;
	//float *d_warpY, *d_warpYs, *d_dwarpY;

	std::vector<float*> pI0;
	std::vector<float*> pI1;
	std::vector<int> pW;
	std::vector<int> pH;
	std::vector<int> pS;
	std::vector<int> pDataSize;
	
	cv::Mat fisheyeMaskPad;
	float* d_fisheyeMask;
	std::vector<float*> pFisheyeMask;
	
	// TGVL1 Process variables
	float *d_a, *d_b, *d_c; // Tensor
	float *d_etau, *d_etav1, *d_etav2;
	float2 *d_p;
	float4 *d_q;
	float *d_u_, *d_u_s, *d_u_last;
	float2 *d_v, *d_vs, *d_v_, *d_v_s;
	float4 *d_gradv;
	float2 *d_Tp;

	// Vector Fields
	cv::Mat translationVector;
	cv::Mat calibrationVector;
	float2 *d_tvForward;
	float2 *d_tvBackward;
	float2 *d_tv2;
	float2 *d_cv;
	float *d_i1calibrated;
	std::vector<float2*> pTvForward;
	std::vector<float2*> pTvBackward;

	// 3D
	float3 *d_X;

	// Debug
	float *debug_depth;

	cv::Mat im0pad, im1pad;


	int initialize(int width, int height, float beta, float gamma,
		float alpha0, float alpha1, float timestep_lambda, float lambda,
		int nLevels, float fScale, int nWarpIters, int nSolverIters);
	int loadVectorFields(cv::Mat translationVector, cv::Mat calibrationVector);

	int copyImagesToDevice(unsigned char* i0, unsigned char* i1);
	int copyImagesToDevice(cv::Mat i0, cv::Mat i1);
	
	int copyMaskToDevice(cv::Mat mask);
	int solveStereoForward();
	int solveStereoForwardMasked();
	int copyStereoToHost(cv::Mat &wCropped);
	int copyStereo8ToHost(cv::Mat& wCropped, float maxDepth);
	int copy1DDisparityToHost(cv::Mat &wCropped);
	int copyDisparityToHost(cv::Mat &wCropped);
	int copyDisparityVisToHost(cv::Mat &wCropped, float flowScale);
	int copyWarpedImageToHost(cv::Mat &wCropped);

	// UTILITIES
	int iAlignUp(int n);
	int iDivUp(int n, int m);
	template<typename T> void Swap(T &a, T &ax);
	template<typename T> void Copy(T &dst, T &src);

	// Kernels
	void FlowToHSV(float2* uv, int w, int h, int s, float3 * uRGB, float flowscale);
	void MedianFilterDisparity(float *inputu,
		int w, int h, int s, float *outputu, int kernelsize);
	void MedianFilter3D(float *X, float *Y, float *Z,
		int w, int h, int s, float *X1, float *Y1, float *Z1, int kernelsize);
	void MedianFilter(float *inputu, float *inputv,
		int w, int h, int s, float *outputu, float*outputv, int kernelsize);
	void LimitRange(float *src, float upperLimit, int w, int h, int s, float *dst);
	void ScalarMultiply(float *src, float scalar, int w, int h, int s);
	//void ScalarMultiply(float2 *src, float scalar, int w, int h, int s);
	void ScalarMultiply(float *src, float scalar, int w, int h, int s, float *dst);
	void ScalarMultiply(float2 *src, float scalar, int w, int h, int s, float2 *dst);
	void Add(float *src1, float* src2, int w, int h, int s, float* dst);
	void Add(float2 *src1, float2* src2, int w, int h, int s, float2* dst);
	void Subtract(float *minuend, float* subtrahend, int w, int h, int s, float* difference);
	void Downscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float *out);
	void Downscale(const float2 *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float2 *out);
	void Downscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void Downscale(const float2 *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float2 *out);
	void Cv8uToGray(uchar * d_iCv8u, float *d_iGray, int w, int h, int s);
	void Cv8uc3ToGray(uchar3 * d_iRgb, float *d_iGray, int w, int h, int s);
	void WarpImage(const float *src, int w, int h, int s,
		const float2 *warpUV, float *out);
	void ComputeOpticalFlowVector(const float *u, const float2 *tv2,
		int w, int h, int s, float2 *warpUV);
	void FindWarpingVector(const float2 *warpUV, const float2 *tv, int w, int h, int s,
		float2 *tv2);
	void FindWarpingVector(const float2 *warpUV, const float *tvx, const float *tvy,
		int w, int h, int s, float2 *tv2);
	void CalcTensor(float* gray, float beta, float gamma, int size_grad,
		int w, int h, int s, float* a, float* b, float* c);
	void Gaussian(float* input, int w, int h, int s, float* output);
	void SolveEta(float alpha0, float alpha1,
		float* a, float *b, float* c,
		int w, int h, int s, float* etau, float* etav1, float* etav2);
	void Clone(float* dst, int w, int h, int s, float* src);
	void Clone(float2* dst, int w, int h, int s, float2* src);
	void ComputeDerivatives(float *I0, float *I1,
		int w, int h, int s, float *Ix, float *Iy, float *Iz);
	void ComputeDerivativesFisheye(float *I0, float *I1, float2 *vector,
		int w, int h, int s, float *Iw, float *Iz);
	void Upscale(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void Upscale(const float2 *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float2 *out);
	void ConvertDisparityToDepth(float *disparity, float baseline, float focal, int w, int h, int s, float *depth);

	void UpdateDualVariablesTGV(float* u_, float2 *v_, float alpha0, float alpha1, float sigma,
		float eta_p, float eta_q, float* a, float* b, float* c,
		int w, int h, int s,
		float4* grad_v, float2* p, float4* q);
	void ThresholdingL1(float2* Tp, float* u_, float* Iu, float* Iz,
		float lambda, float tau, float* eta_u, float* u, float* us,
		int w, int h, int s);
	void UpdatePrimalVariables(float* u_, float2* v_, float2* p, float4* q,
		float* a, float* b, float* c,
		float tau, float* eta_v1, float* eta_v2,
		float alpha0, float alpha1, float mu,
		float* u, float2* v,
		float* u_s, float2* v_s,
		int w, int h, int s);
	void SolveTp(float* a, float* b, float* c, float2* p, 
		int w, int h, int s, float2* Tp);

	// Kernels Masked version
	void GaussianMasked(float* input, float* mask, int w, int h, int s, float* output);
	void CalcTensorMasked(float* gray, float* mask, float beta, float gamma, int size_grad,
		int w, int h, int s, float* a, float* b, float* c);
	void ComputeDerivativesFisheyeMasked(float *I0, float *I1, float2 *vector, float* mask,
		int w, int h, int s, float *Iw, float *Iz);
	void ConvertDisparityToDepthMasked(float *disparity, float* mask, float baseline, float focal,
		int w, int h, int s, float *depth);
	void DownscaleNearestNeighbor(const float *src, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float *out);
	void DownscaleMasked(const float *src, float *mask, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float *out);
	void DownscaleMasked(const float2 *src, float * mask, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float2 *out);
	void DownscaleMasked(const float *src, float* mask, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void DownscaleMasked(const float2 *src, float * mask, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float2 *out);
	void MedianFilterDisparityMasked(float *inputu, float* mask,
		int w, int h, int s, float *outputu, int kernelsize);
	void SolveEtaMasked(float* mask, float alpha0, float alpha1,
		float* a, float *b, float* c, int w, int h, int s, float* etau, float* etav1, float* etav2);
	void UpscaleMasked(const float *src, float* mask, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float *out);
	void UpscaleMasked(const float2 *src, float * mask, int width, int height, int stride,
		int newWidth, int newHeight, int newStride, float scale, float2 *out);
	void ComputeOpticalFlowVectorMasked(const float *u, const float2 *tv2, float* mask,
		int w, int h, int s, float2 *warpUV);
	void FindWarpingVectorMasked(const float2 *warpUV, float* mask, const float2 *tv,
		int w, int h, int s, float2 *tv2);
	void FindWarpingVectorMasked(const float2 *warpUV, float* mask, const float *tvx, const float *tvy,
		int w, int h, int s, float2 *tv2);
	void WarpImageMasked(const float *src, float* mask, int w, int h, int s,
		const float2 *warpUV, float *out);
	void ThresholdingL1Masked(float2* Tp, float* u_, float* Iu, float* Iz, float * mask,
		float lambda, float tau, float* eta_u, float* u, float* us,
		int w, int h, int s);
	void SolveTpMasked(float* mask, float* a, float* b, float* c, float2* p,
		int w, int h, int s, float2* Tp);
	void UpdateDualVariablesTGVMasked(float* mask, float* u_, float2 *v_, float alpha0, float alpha1, float sigma,
		float eta_p, float eta_q,
		float* a, float* b, float* c,
		int w, int h, int s,
		float4* grad_v, float2* p, float4* q);
	void UpdatePrimalVariablesMasked(float * mask, float* u_, float2* v_, float2* p, float4* q,
		float* a, float* b, float* c,
		float tau, float* eta_v1, float* eta_v2,
		float alpha0, float alpha1, float mu,
		float* u, float2* v,
		float* u_s, float2* v_s,
		int w, int h, int s);
};