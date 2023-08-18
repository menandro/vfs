#include "stereotgv.h"

void DEBUGIMAGE(std::string windowName, float* deviceImage, int height, int stride, bool verbose, bool wait) {
	cv::Mat calibrated = cv::Mat(height, stride, CV_32F);
	checkCudaErrors(cudaMemcpy((float *)calibrated.ptr(), deviceImage, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	cv::imshow(windowName, calibrated);
	if (verbose) {
		std::cout << windowName << " " << calibrated.at<float>(height / 2, stride / 2) << std::endl;
	}
	if (wait) {
		cv::waitKey();
	}
	
}

void DEBUGIMAGE(std::string windowName, float2* deviceImage, int height, int stride, bool verbose, bool wait) {
	cv::Mat calibrated = cv::Mat(height, stride, CV_32FC2);
	checkCudaErrors(cudaMemcpy((float2 *)calibrated.ptr(), deviceImage, stride * height * sizeof(float2), cudaMemcpyDeviceToHost));
	cv::Mat cuv[2];
	cv::split(calibrated, cuv);
	cv::imshow(windowName+"0", cuv[0]);
	cv::imshow(windowName+"1", cuv[1]);
	if (verbose) {
		std::cout << windowName << " " << cuv[0].at<float>(height / 2, stride - 40) << std::endl;
	}
	if (wait) {
		cv::waitKey();
	}
}


StereoTgv::StereoTgv() {
	this->BlockHeight = 1;
	this->BlockWidth = 32;
	this->StrideAlignment = 32;
}

StereoTgv::StereoTgv(int blockWidth, int blockHeight, int strideAlignment) {
	this->BlockHeight = blockHeight;
	this->BlockWidth = blockWidth;
	this->StrideAlignment = strideAlignment;
}

int StereoTgv::initialize(int width, int height, float beta, float gamma,
	float alpha0, float alpha1, float timestep_lambda, float lambda,
	int nLevels, float fScale, int nWarpIters, int nSolverIters) {
	// Set memory for lidarinput (32fc1), lidarmask(32fc1), image0, image1 (8uc3), depthout (32fc1)
	// flowinput (32fc2), depthinput (32fc1)
	this->width = width;
	this->height = height;
	this->stride = this->iAlignUp(width);

	this->beta = beta;
	this->gamma = gamma;
	this->alpha0 = alpha0;
	this->alpha1 = alpha1;
	this->timestep_lambda = timestep_lambda;
	this->lambda = lambda;
	this->fScale = fScale;
	this->nLevels = nLevels;
	this->nWarpIters = nWarpIters;
	this->nSolverIters = nSolverIters;

	pI0 = std::vector<float*>(nLevels);
	pI1 = std::vector<float*>(nLevels);
	pW = std::vector<int>(nLevels);
	pH = std::vector<int>(nLevels);
	pS = std::vector<int>(nLevels);
	pDataSize = std::vector<int>(nLevels);
	pTvForward = std::vector<float2*>(nLevels);
	pTvBackward = std::vector<float2*>(nLevels);
	pFisheyeMask = std::vector<float*>(nLevels);

	int newHeight = height;
	int newWidth = width;
	int newStride = iAlignUp(width);
	//std::cout << "Pyramid Sizes: " << newWidth << " " << newHeight << " " << newStride << std::endl;
	for (int level = 0; level < nLevels; level++) {
		pDataSize[level] = newStride * newHeight * sizeof(float);
		checkCudaErrors(cudaMalloc(&pI0[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pI1[level], pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pTvForward[level], 2 * pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pTvBackward[level], 2 * pDataSize[level]));
		checkCudaErrors(cudaMalloc(&pFisheyeMask[level], pDataSize[level]));

		//std::cout << newHeight << " " << newWidth << " " << newStride << std::endl;

		pW[level] = newWidth;
		pH[level] = newHeight;
		pS[level] = newStride;
		newHeight = (int)((float)newHeight / fScale);
		newWidth = (int)((float)newWidth / fScale);
		newStride = iAlignUp(newWidth);
		
	}

	//std::cout << stride << " " << height << std::endl;
	dataSize8u = stride * height * sizeof(uchar);
	dataSize8uc3 = stride * height * sizeof(uchar3);
	dataSize32f = stride * height * sizeof(float);
	dataSize32fc2 = stride * height * sizeof(float2);
	dataSize32fc3 = stride * height * sizeof(float3);
	dataSize32fc4 = stride * height * sizeof(float4);

	// Inputs and Outputs
	checkCudaErrors(cudaMalloc(&d_i0, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i1, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i1warp, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i08u, dataSize8u));
	checkCudaErrors(cudaMalloc(&d_i18u, dataSize8u));
	checkCudaErrors(cudaMalloc(&d_i08uc3, dataSize8uc3));
	checkCudaErrors(cudaMalloc(&d_i18uc3, dataSize8uc3));
	checkCudaErrors(cudaMalloc(&d_i0smooth, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_i1smooth, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_Iu, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_Iz, dataSize32f));
	// Output Disparity
	checkCudaErrors(cudaMalloc(&d_u, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_du, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_us, dataSize32f));
	// Output Depth
	checkCudaErrors(cudaMalloc(&d_depth, dataSize32f));
	// Warping Variables
	checkCudaErrors(cudaMalloc(&d_warpUV, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_dwarpUV, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_warpUVs, dataSize32fc2));

	// Vector Fields
	checkCudaErrors(cudaMalloc(&d_tvForward, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_tvBackward, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_tv2, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_cv, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_i1calibrated, dataSize32f));

	// Process variables
	checkCudaErrors(cudaMalloc(&d_a, dataSize32f)); // Tensor
	checkCudaErrors(cudaMalloc(&d_b, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_c, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_etau, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_etav1, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_etav2, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_p, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_q, dataSize32fc4));
	
	checkCudaErrors(cudaMalloc(&d_u_, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_u_last, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_u_s, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_v, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_vs, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_v_, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_v_s, dataSize32fc2));
	checkCudaErrors(cudaMalloc(&d_gradv, dataSize32fc4));
	checkCudaErrors(cudaMalloc(&d_Tp, dataSize32fc2));

	// 3D
	checkCudaErrors(cudaMalloc(&d_X, dataSize32fc3));
	X = cv::Mat(height, stride, CV_32FC3);

	// Debugging
	checkCudaErrors(cudaMalloc(&debug_depth, dataSize32f));
	checkCudaErrors(cudaMalloc(&d_uvrgb, dataSize32fc3));

	depth = cv::Mat(height, stride, CV_32F);
	warpUV = cv::Mat(height, stride, CV_32FC2);
	warpUVrgb = cv::Mat(height, stride, CV_32FC3);

	return 0;
}


int StereoTgv::loadVectorFields(cv::Mat translationVector, cv::Mat calibrationVector) {
	// Padding
	cv::Mat translationVectorPad = cv::Mat(height, stride, CV_32FC2);
	cv::Mat calibrationVectorPad = cv::Mat(height, stride, CV_32FC2);
	cv::copyMakeBorder(translationVector, translationVectorPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(calibrationVector, calibrationVectorPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	// Translation Vector Field
	//translationVector = cv::Mat(height, stride, CV_32FC2);
	//calibrationVector = cv::Mat(height, stride, CV_32FC2);

	checkCudaErrors(cudaMemcpy(d_tvForward, (float2 *)translationVectorPad.ptr(), dataSize32fc2, cudaMemcpyHostToDevice));

	ScalarMultiply(d_tvForward, -1.0f, width, height, stride, d_tvBackward);
	Swap(pTvForward[0], d_tvForward);
	Swap(pTvBackward[0], d_tvBackward);
	for (int level = 1; level < nLevels; level++) {
		//std::cout << "vectorfields " << pW[level] << " " << pH[level] << " " << pS[level] << std::endl;
		Downscale(pTvForward[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pTvForward[level]);
		Downscale(pTvBackward[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pTvBackward[level]);
		
	}

	// Calibration Vector Field
	checkCudaErrors(cudaMemcpy(d_cv, (float2 *)calibrationVectorPad.ptr(), dataSize32fc2, cudaMemcpyHostToDevice));
	return 0;
}

int StereoTgv::copyImagesToDevice(unsigned char* i0data, unsigned char* i1data) {
	cv::Mat i0 = cv::Mat(cv::Size(width, height), CV_8UC1, i0data);
	cv::Mat i1 = cv::Mat(cv::Size(width, height), CV_8UC1, i1data);
	cv::copyMakeBorder(i0, im0pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(i1, im1pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	checkCudaErrors(cudaMemcpy(d_i08u, (uchar*)im0pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_i18u, (uchar*)im1pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
	// Convert to 32F
	Cv8uToGray(d_i08u, pI0[0], width, height, stride);
	Cv8uToGray(d_i18u, pI1[0], width, height, stride);
	return 0;
}

int StereoTgv::copyImagesToDevice(cv::Mat i0, cv::Mat i1) {
	// Padding
	cv::copyMakeBorder(i0, im0pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	cv::copyMakeBorder(i1, im1pad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);

	if (i0.type() == CV_8U) {
		checkCudaErrors(cudaMemcpy(d_i08u, (uchar *)im0pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18u, (uchar *)im1pad.ptr(), dataSize8u, cudaMemcpyHostToDevice));
		// Convert to 32F
		Cv8uToGray(d_i08u, pI0[0], width, height, stride);
		Cv8uToGray(d_i18u, pI1[0], width, height, stride);
	}
	else if (i0.type() == CV_32F) {
		checkCudaErrors(cudaMemcpy(pI0[0], (float *)im0pad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(pI1[0], (float *)im1pad.ptr(), dataSize32f, cudaMemcpyHostToDevice));
	}
	else if (i0.type() == CV_8UC3){
		checkCudaErrors(cudaMemcpy(d_i08uc3, (uchar3 *)im0pad.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_i18uc3, (uchar3 *)im1pad.ptr(), dataSize8uc3, cudaMemcpyHostToDevice));
		// Convert to 32F
		Cv8uc3ToGray(d_i08uc3, pI0[0], width, height, stride);
		Cv8uc3ToGray(d_i18uc3, pI1[0], width, height, stride);
	}
	return 0;
}

int StereoTgv::copyMaskToDevice(cv::Mat mask) {
	cv::copyMakeBorder(mask, fisheyeMaskPad, 0, 0, 0, stride - width, cv::BORDER_CONSTANT, 0);
	checkCudaErrors(cudaMemcpy(pFisheyeMask[0], (float *)fisheyeMaskPad.ptr(), dataSize32f, cudaMemcpyHostToDevice));

	for (int level = 1; level < nLevels; level++) {
		//std::cout << pW[level] << " " << pH[level] << " " << pS[level] << std::endl;
		DownscaleNearestNeighbor(pFisheyeMask[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pFisheyeMask[level]);
		//DEBUGIMAGE("maskasdfadf", pFisheyeMask[level], pH[level], pS[level], true, true);
	}
	return 0;
}

bool isFirstFrame = true;

int StereoTgv::solveStereoForward() {
	// Warp i1 using vector fields
	WarpImage(pI1[0], width, height, stride, d_cv, d_i1calibrated);
	Swap(pI1[0], d_i1calibrated);
	//DEBUGIMAGE("i1", pI1[0], height, stride, true, false);
	
	checkCudaErrors(cudaMemset(d_u, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_u_, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_warpUV, 0, dataSize32fc2));

	// Construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level],pI0[level]);
		Downscale(pI1[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI1[level]);
	}

	// Solve stereo
	for (int level = nLevels - 1; level >= 0; level--) {
		checkCudaErrors(cudaMemset(d_a, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_b, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_c, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_etau, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_etav1, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_etav2, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_i0smooth, 0, dataSize32f));
		float eta_p = 3.0f;
		float eta_q = 2.0f;

		if (level == nLevels - 1) {
			ComputeOpticalFlowVector(d_u, pTvForward[level], pW[level], pH[level], pS[level], d_warpUV);
		}

		// Calculate anisotropic diffucion tensor
		Gaussian(pI0[level], pW[level], pH[level], pS[level], d_i0smooth);
		CalcTensor(d_i0smooth, beta, gamma, 2, pW[level], pH[level], pS[level], d_a, d_b, d_c);
		SolveEta(alpha0, alpha1, d_a, d_b, d_c, 
			pW[level], pH[level], pS[level], d_etau, d_etav1, d_etav2);

		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
			checkCudaErrors(cudaMemset(d_q, 0, dataSize32fc4));
			checkCudaErrors(cudaMemset(d_v, 0, dataSize32fc2));
			Clone(d_v_, pW[level], pH[level], pS[level], d_v);
			checkCudaErrors(cudaMemset(d_gradv, 0, dataSize32fc4));
			checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));
			/*checkCudaErrors(cudaMemset(d_Tp, 0, dataSize32fc2));
			checkCudaErrors(cudaMemset(d_Iu, 0, dataSize32f));
			checkCudaErrors(cudaMemset(d_Iz, 0, dataSize32f));
			checkCudaErrors(cudaMemset(d_i1warp, 0, dataSize32f));*/

			FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);

			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);
			if (level == 0) {
				DEBUGIMAGE("i0", pI0[level], pH[level], pS[level], false, false);
				//DEBUGIMAGE("i1", pI1[level], pH[level], pS[level], false, false);
				DEBUGIMAGE("iwarp", d_i1warp, pH[level], pS[level], false, false);
			}
			
			
			ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level], 
				pW[level], pH[level], pS[level], d_Iu, d_Iz);
			Clone(d_u_last, pW[level], pH[level], pS[level], d_u);
			
			float tau = 1.0f;
			float sigma = 1.0f / tau;

			// Inner iteration
			for (int iter = 0; iter < nSolverIters; iter++) {
				float mu;
				if (sigma < 1000.0f) mu = 1.0f / sqrt(1 + 0.7f * tau * timestep_lambda);
				else mu = 1;

				// Solve Dual Variables
				UpdateDualVariablesTGV(d_u_, d_v_, alpha0, alpha1, sigma, eta_p, eta_q,
					d_a, d_b, d_c, pW[level], pH[level], pS[level],
					 d_gradv, d_p, d_q);

				// Solve Thresholding
				SolveTp(d_a, d_b, d_c, d_p, pW[level], pH[level], pS[level], d_Tp);
				ThresholdingL1(d_Tp, d_u_, d_Iu, d_Iz, lambda, tau, d_etau, d_u, d_us, pW[level], pH[level], pS[level]);
				Swap(d_u, d_us);
				
				// Solve Primal Variables
				UpdatePrimalVariables(d_u_, d_v_, d_p, d_q, d_a, d_b, d_c, tau, d_etav1, d_etav2,
					alpha0, alpha1, mu, d_u, d_v, d_u_s, d_v_s, pW[level], pH[level], pS[level]);
				Swap(d_u_, d_u_s);
				Swap(d_v_, d_v_s);

				sigma = sigma / mu;
				tau = tau * mu;
			}
			/*MedianFilterDisparity(d_u, pW[level], pH[level], pS[level], d_us, 5);
			Swap(d_u, d_us);*/

			// Calculate d_warpUV
			Subtract(d_u, d_u_last, pW[level], pH[level], pS[level], d_du);

			// Sanity Check
			LimitRange(d_du, limitRange, pW[level], pH[level], pS[level], d_du);

			Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u);
			Clone(d_u_, pW[level], pH[level], pS[level], d_u);

			ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
			Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);
		}

		// Upscale
		if (level > 0)
		{
			float scale = fScale;
			Upscale(d_u, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_u_, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_u_s);
			Upscale(d_warpUV, pW[level], pH[level], pS[level], pW[level - 1], pH[level - 1], pS[level - 1], scale, d_warpUVs);
			Swap(d_u, d_us);
			Swap(d_u_, d_u_s);
			Swap(d_warpUV, d_warpUVs);
		}
		isFirstFrame = false;
	}

	/*Clone(d_w, width, height, stride, d_wForward);

	if (visualizeResults) {
		FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
	}*/

	return 0;
}

int StereoTgv::solveStereoForwardMasked() {
	// Warp i1 using vector fields
	//WarpImageMasked(pI1[0], pFisheyeMask[0], width, height, stride, d_cv, d_i1calibrated);
	WarpImage(pI1[0], width, height, stride, d_cv, d_i1calibrated);
	Swap(pI1[0], d_i1calibrated);

	checkCudaErrors(cudaMemset(d_u, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_u_, 0, dataSize32f));
	checkCudaErrors(cudaMemset(d_warpUV, 0, dataSize32fc2));

	// Construct pyramid
	for (int level = 1; level < nLevels; level++) {
		Downscale(pI0[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI0[level]);
		Downscale(pI1[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
			pW[level], pH[level], pS[level], pI1[level]);
	}

	// Solve stereo
	for (int level = nLevels - 1; level >= 0; level--) {
		checkCudaErrors(cudaMemset(d_a, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_b, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_c, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_etau, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_etav1, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_etav2, 0, dataSize32f));
		checkCudaErrors(cudaMemset(d_i0smooth, 0, dataSize32f));
		float eta_p = 3.0f;
		float eta_q = 2.0f;

		if (level == nLevels - 1) {
			ComputeOpticalFlowVector(d_u, pTvForward[level], pW[level], pH[level], pS[level], d_warpUV);
		}

		// Calculate anisotropic diffucion tensor
		GaussianMasked(pI0[level], pFisheyeMask[level], pW[level], pH[level], pS[level], d_i0smooth);
		CalcTensorMasked(d_i0smooth, pFisheyeMask[level], beta, gamma, 2, pW[level], pH[level], pS[level], d_a, d_b, d_c);
		SolveEtaMasked(pFisheyeMask[level], alpha0, alpha1, d_a, d_b, d_c,
			pW[level], pH[level], pS[level], d_etau, d_etav1, d_etav2);
		//DEBUGIMAGE("mask", pFisheyeMask[level], pH[level], pS[level], false, true);

		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
			checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
			checkCudaErrors(cudaMemset(d_q, 0, dataSize32fc4));
			checkCudaErrors(cudaMemset(d_v, 0, dataSize32fc2));
			Clone(d_v_, pW[level], pH[level], pS[level], d_v);
			checkCudaErrors(cudaMemset(d_gradv, 0, dataSize32fc4));
			checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));

			FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);

			/*ComputeDerivativesFisheyeMasked(pI0[level], d_i1warp, pTvForward[level], pFisheyeMask[level],
				pW[level], pH[level], pS[level], d_Iu, d_Iz);*/
			ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level],
				pW[level], pH[level], pS[level], d_Iu, d_Iz);

			//DEBUGIMAGE("mask2", pI1[level], pH[level], pS[level], false, true);
			Clone(d_u_last, pW[level], pH[level], pS[level], d_u);

			float tau = 1.0f;
			float sigma = 1.0f / tau;

			// Inner iteration
			for (int iter = 0; iter < nSolverIters; iter++) {
				float mu;
				if (sigma < 1000.0f) mu = 1.0f / sqrt(1 + 0.7f * tau * timestep_lambda);
				else mu = 1;

				// Solve Dual Variables
				UpdateDualVariablesTGVMasked(pFisheyeMask[level], d_u_, d_v_, alpha0, alpha1, sigma, eta_p, eta_q,
					d_a, d_b, d_c, pW[level], pH[level], pS[level],
					d_gradv, d_p, d_q);

				// Solve Thresholding
				SolveTpMasked(pFisheyeMask[level], d_a, d_b, d_c, d_p, pW[level], pH[level], pS[level], d_Tp);
				//SolveTp(d_a, d_b, d_c, d_p, pW[level], pH[level], pS[level], d_Tp);
				ThresholdingL1Masked(d_Tp, d_u_, d_Iu, d_Iz, pFisheyeMask[level], lambda, tau, d_etau, d_u, d_us,
					pW[level], pH[level], pS[level]);
				Swap(d_u, d_us);

				// Solve Primal Variables
				UpdatePrimalVariablesMasked(pFisheyeMask[level], d_u_, d_v_, d_p, d_q, d_a, d_b, d_c, tau, d_etav1, d_etav2,
					alpha0, alpha1, mu, d_u, d_v, d_u_s, d_v_s, pW[level], pH[level], pS[level]);
				Swap(d_u_, d_u_s);
				Swap(d_v_, d_v_s);

				sigma = sigma / mu;
				tau = tau * mu;
			}

			// Calculate d_warpUV
			MedianFilterDisparity(d_u, pW[level], pH[level], pS[level], d_us, 5);
			Swap(d_u, d_us);

			Subtract(d_u, d_u_last, pW[level], pH[level], pS[level], d_du);

			// Sanity Check (ICRA2020)
			LimitRange(d_du, limitRange, pW[level], pH[level], pS[level], d_du);

			Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u);
			Clone(d_u_, pW[level], pH[level], pS[level], d_u);

			ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
			Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);
		}

		// Upscale
		if (level > 0)
		{
			float scale = fScale;
			Upscale(d_u, pW[level], pH[level], pS[level], 
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
			Upscale(d_u_, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_u_s);
			Upscale(d_warpUV, pW[level], pH[level], pS[level],
				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_warpUVs);

			Swap(d_u, d_us);
			Swap(d_u_, d_u_s);
			Swap(d_warpUV, d_warpUVs);
		}
		isFirstFrame = false;
	}

	return 0;
}

int StereoTgv::copyStereoToHost(cv::Mat& croppedDepth, cv::Mat& croppedX, float focalx, float focaly,
	float cx, float cy, float d1, float d2, float d3, float d4,
	float t1, float t2, float t3) {
	// Convert Disparity to Depth/3D
	ConvertKB(d_warpUV, focalx, focaly, cx, cy, d1, d2, d3, d4, t1, t2, t3, d_X, d_depth, width, height, stride);

	// Remove Padding
	checkCudaErrors(cudaMemcpy((float*)depth.ptr(), d_depth, dataSize32f, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	croppedDepth = depth(roi);

	checkCudaErrors(cudaMemcpy((float3*)X.ptr(), d_X, dataSize32fc3, cudaMemcpyDeviceToHost));
	croppedX = X(roi);
	return 0;
}

int StereoTgv::copy1DDisparityToHost(cv::Mat &wCropped) {
	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_u, dataSize32f, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = depth(roi);
	return 0;
}

int StereoTgv::copyDisparityToHost(cv::Mat &wCropped) {
	// Remove Padding
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float2 *)warpUV.ptr(), d_warpUV, dataSize32fc2, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = warpUV(roi);
	return 0;
}

int StereoTgv::copyDisparityVisToHost(cv::Mat &wCropped, float flowScale) {
	// Remove Padding
	FlowToHSV(d_warpUV, width, height, stride, d_uvrgb, flowScale);
	checkCudaErrors(cudaMemcpy((float3 *)warpUVrgb.ptr(), d_uvrgb, dataSize32fc3, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = warpUVrgb(roi);
	std::cout << height << " " << width << std::endl;
	return 0;
}

int StereoTgv::copyWarpedImageToHost(cv::Mat &wCropped) {
	cv::Mat warpedImage = cv::Mat(cv::Size(stride, height), CV_32F);
	//checkCudaErrors(cudaMemcpy((float *)depth.ptr(), d_w, stride * height * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((float *)warpedImage.ptr(), d_i1warp, dataSize32f, cudaMemcpyDeviceToHost));
	cv::Rect roi(0, 0, width, height); // define roi here as x0, y0, width, height
	wCropped = warpedImage(roi);
	return 0;
}

// Utilities
int StereoTgv::iAlignUp(int n)
{
	int m = this->StrideAlignment;
	int mod = n % m;

	if (mod)
		return n + m - mod;
	else
		return n;
}

int StereoTgv::iDivUp(int n, int m)
{
	return (n + m - 1) / m;
}

template<typename T> void StereoTgv::Swap(T &a, T &ax)
{
	T t = a;
	a = ax;
	ax = t;
}

template<typename T> void StereoTgv::Copy(T &dst, T &src)
{
	dst = src;
}











//// Working TGVL1 with masking
//int StereoTgv::solveStereoForwardMasked() {
//	// Warp i1 using vector fields
//	//WarpImageMasked(pI1[0], pFisheyeMask[0], width, height, stride, d_cv, d_i1calibrated);
//	WarpImage(pI1[0], width, height, stride, d_cv, d_i1calibrated);
//	Swap(pI1[0], d_i1calibrated);
//	//DEBUGIMAGE("i1", pI1[0], height, stride, true, false);
//
//	checkCudaErrors(cudaMemset(d_u, 0, dataSize32f));
//	checkCudaErrors(cudaMemset(d_u_, 0, dataSize32f));
//	checkCudaErrors(cudaMemset(d_warpUV, 0, dataSize32fc2));
//
//	// Construct pyramid
//	for (int level = 1; level < nLevels; level++) {
//		/*DownscaleMasked(pI0[level - 1], pFisheyeMask[0], pW[level - 1], pH[level - 1], pS[level - 1],
//			pW[level], pH[level], pS[level], pI0[level]);
//		DownscaleMasked(pI1[level - 1], pFisheyeMask[0], pW[level - 1], pH[level - 1], pS[level - 1],
//			pW[level], pH[level], pS[level], pI1[level]);*/
//		Downscale(pI0[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
//			pW[level], pH[level], pS[level], pI0[level]);
//		Downscale(pI1[level - 1], pW[level - 1], pH[level - 1], pS[level - 1],
//			pW[level], pH[level], pS[level], pI1[level]);
//		//std::cout << pH[level] << " " << pW[level] << " " << pS[level] << std::endl;
//	}
//
//	// Solve stereo
//	for (int level = nLevels - 1; level >= 0; level--) {
//		checkCudaErrors(cudaMemset(d_a, 0, dataSize32f));
//		checkCudaErrors(cudaMemset(d_b, 0, dataSize32f));
//		checkCudaErrors(cudaMemset(d_c, 0, dataSize32f));
//		checkCudaErrors(cudaMemset(d_etau, 0, dataSize32f));
//		checkCudaErrors(cudaMemset(d_etav1, 0, dataSize32f));
//		checkCudaErrors(cudaMemset(d_etav2, 0, dataSize32f));
//		checkCudaErrors(cudaMemset(d_i0smooth, 0, dataSize32f));
//		float eta_p = 3.0f;
//		float eta_q = 2.0f;
//
//		if (level == nLevels - 1) {
//			//ComputeOpticalFlowVectorMasked(d_u, pTvForward[level], pFisheyeMask[level], pW[level], pH[level], pS[level], d_warpUV);
//			ComputeOpticalFlowVector(d_u, pTvForward[level], pW[level], pH[level], pS[level], d_warpUV);
//		}
//
//		// Calculate anisotropic diffucion tensor
//		GaussianMasked(pI0[level], pFisheyeMask[level], pW[level], pH[level], pS[level], d_i0smooth);
//		//Gaussian(pI0[level], pW[level], pH[level], pS[level], d_i0smooth);
//		CalcTensorMasked(d_i0smooth, pFisheyeMask[level], beta, gamma, 2, pW[level], pH[level], pS[level], d_a, d_b, d_c);
//		//CalcTensor(d_i0smooth, beta, gamma, 2, pW[level], pH[level], pS[level], d_a, d_b, d_c);
//		SolveEtaMasked(pFisheyeMask[level], alpha0, alpha1, d_a, d_b, d_c,
//			pW[level], pH[level], pS[level], d_etau, d_etav1, d_etav2);
//		/*SolveEta(alpha0, alpha1, d_a, d_b, d_c,
//			pW[level], pH[level], pS[level], d_etau, d_etav1, d_etav2);*/
//
//		for (int warpIter = 0; warpIter < nWarpIters; warpIter++) {
//			checkCudaErrors(cudaMemset(d_p, 0, dataSize32fc2));
//			checkCudaErrors(cudaMemset(d_q, 0, dataSize32fc4));
//			checkCudaErrors(cudaMemset(d_v, 0, dataSize32fc2));
//			Clone(d_v_, pW[level], pH[level], pS[level], d_v);
//			checkCudaErrors(cudaMemset(d_gradv, 0, dataSize32fc4));
//			checkCudaErrors(cudaMemset(d_du, 0, dataSize32f));
//			/*checkCudaErrors(cudaMemset(d_Tp, 0, dataSize32fc2));
//			checkCudaErrors(cudaMemset(d_Iu, 0, dataSize32f));
//			checkCudaErrors(cudaMemset(d_Iz, 0, dataSize32f));
//			checkCudaErrors(cudaMemset(d_i1warp, 0, dataSize32f));*/
//
//			//FindWarpingVectorMasked(d_warpUV, pFisheyeMask[level], pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
//			FindWarpingVector(d_warpUV, pTvForward[level], pW[level], pH[level], pS[level], d_tv2);
//
//			//WarpImageMasked(pI1[level], pFisheyeMask[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);
//			WarpImage(pI1[level], pW[level], pH[level], pS[level], d_warpUV, d_i1warp);
//			if (level == level) {
//				std::cout << "iter: " << pH[level] << " " << pS[level] << std::endl;
//				DEBUGIMAGE("i0", pI0[level], pH[level], pS[level], false, false);
//				//DEBUGIMAGE("i1", pI1[level], pH[level], pS[level], false, false);
//				DEBUGIMAGE("iwarp", d_i1warp, pH[level], pS[level], false, false);
//				cv::waitKey(1);
//			}
//
//
//			ComputeDerivativesFisheyeMasked(pI0[level], d_i1warp, pTvForward[level], pFisheyeMask[level],
//				pW[level], pH[level], pS[level], d_Iu, d_Iz);
//			/*ComputeDerivativesFisheye(pI0[level], d_i1warp, pTvForward[level],
//				pW[level], pH[level], pS[level], d_Iu, d_Iz);*/
//			Clone(d_u_last, pW[level], pH[level], pS[level], d_u);
//
//			float tau = 1.0f;
//			float sigma = 1.0f / tau;
//
//			// Inner iteration
//			for (int iter = 0; iter < nSolverIters; iter++) {
//				float mu;
//				if (sigma < 1000.0f) mu = 1.0f / sqrt(1 + 0.7f * tau * timestep_lambda);
//				else mu = 1;
//
//				// Solve Dual Variables
//				UpdateDualVariablesTGVMasked(pFisheyeMask[level], d_u_, d_v_, alpha0, alpha1, sigma, eta_p, eta_q,
//					d_a, d_b, d_c, pW[level], pH[level], pS[level],
//					d_gradv, d_p, d_q);
//				/*UpdateDualVariablesTGV(d_u_, d_v_, alpha0, alpha1, sigma, eta_p, eta_q,
//					d_a, d_b, d_c, pW[level], pH[level], pS[level],
//					d_gradv, d_p, d_q);*/
//
//					// Solve Thresholding
//				SolveTpMasked(pFisheyeMask[level], d_a, d_b, d_c, d_p, pW[level], pH[level], pS[level], d_Tp);
//				//SolveTp(d_a, d_b, d_c, d_p, pW[level], pH[level], pS[level], d_Tp);
//				ThresholdingL1Masked(d_Tp, d_u_, d_Iu, d_Iz, pFisheyeMask[level], lambda, tau, d_etau, d_u, d_us,
//					pW[level], pH[level], pS[level]);
//				/*ThresholdingL1(d_Tp, d_u_, d_Iu, d_Iz, lambda, tau, d_etau, d_u, d_us,
//					pW[level], pH[level], pS[level]);*/
//				Swap(d_u, d_us);
//
//				// Solve Primal Variables
//				UpdatePrimalVariablesMasked(pFisheyeMask[level], d_u_, d_v_, d_p, d_q, d_a, d_b, d_c, tau, d_etav1, d_etav2,
//					alpha0, alpha1, mu, d_u, d_v, d_u_s, d_v_s, pW[level], pH[level], pS[level]);
//				/*UpdatePrimalVariables(d_u_, d_v_, d_p, d_q, d_a, d_b, d_c, tau, d_etav1, d_etav2,
//					alpha0, alpha1, mu, d_u, d_v, d_u_s, d_v_s, pW[level], pH[level], pS[level]);*/
//				Swap(d_u_, d_u_s);
//				Swap(d_v_, d_v_s);
//
//				sigma = sigma / mu;
//				tau = tau * mu;
//			}
//			/*MedianFilterDisparity(d_u, pW[level], pH[level], pS[level], d_us, 5);
//			Swap(d_u, d_us);*/
//
//			// Calculate d_warpUV
//			Subtract(d_u, d_u_last, pW[level], pH[level], pS[level], d_du);
//
//			// Sanity Check
//			LimitRange(d_du, 1.0f, pW[level], pH[level], pS[level], d_du);
//
//			Add(d_u_last, d_du, pW[level], pH[level], pS[level], d_u);
//			Clone(d_u_, pW[level], pH[level], pS[level], d_u);
//
//			//ComputeOpticalFlowVectorMasked(d_du, d_tv2, pFisheyeMask[level], pW[level], pH[level], pS[level], d_dwarpUV);
//			ComputeOpticalFlowVector(d_du, d_tv2, pW[level], pH[level], pS[level], d_dwarpUV);
//			Add(d_warpUV, d_dwarpUV, pW[level], pH[level], pS[level], d_warpUV);
//		}
//
//		// Upscale
//		if (level > 0)
//		{
//			float scale = fScale;
//			/*UpscaleMasked(d_u, pFisheyeMask[level], pW[level], pH[level], pS[level],
//				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);*/
//			Upscale(d_u, pW[level], pH[level], pS[level],
//				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_us);
//			/*UpscaleMasked(d_u_, pFisheyeMask[level], pW[level], pH[level], pS[level],
//				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_u_s);*/
//			Upscale(d_u_, pW[level], pH[level], pS[level],
//				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_u_s);
//			/*UpscaleMasked(d_warpUV, pFisheyeMask[level], pW[level], pH[level], pS[level],
//				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_warpUVs);*/
//			Upscale(d_warpUV, pW[level], pH[level], pS[level],
//				pW[level - 1], pH[level - 1], pS[level - 1], scale, d_warpUVs);
//
//			Swap(d_u, d_us);
//			Swap(d_u_, d_u_s);
//			Swap(d_warpUV, d_warpUVs);
//		}
//		isFirstFrame = false;
//	}
//
//	/*Clone(d_w, width, height, stride, d_wForward);
//
//	if (visualizeResults) {
//		FlowToHSV(d_u, d_v, width, height, stride, d_uvrgb, flowScale);
//	}*/
//
//	return 0;
//}
