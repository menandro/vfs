#include "vfs.h"

extern "C" __declspec(dllexport) StereoTgv * Create() {
	StereoTgv* vfs = new StereoTgv();
	return vfs;
}

extern "C" __declspec(dllexport) void Init(StereoTgv * vfs, int stereoWidth, int stereoHeight, float beta, float gamma,
	float alpha0, float alpha1, float timeStepLambda, float lambda, int nLevel, float fScale,
	int nWarpIters, int nSolverIters, float focal, float baseline, const char* fisheyeMaskFilename, const char* translationVectorFilename,
	const char* calibrationVectorFilename) {
	// Initialize
	vfs->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	vfs->baseline = baseline; 
	vfs->focal = focal;
	
	// Load Fisheye mask
	cv::Mat fisheyeMask8 = cv::imread(fisheyeMaskFilename, cv::IMREAD_GRAYSCALE);
	cv::Mat fisheyeMask;
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	vfs->copyMaskToDevice(fisheyeMask);

	// Load Vector Fields
	cv::Mat translationVector, calibrationVector;
	translationVector = cv::readOpticalFlow(translationVectorFilename);
	calibrationVector = cv::readOpticalFlow(calibrationVectorFilename);
	vfs->loadVectorFields(translationVector, calibrationVector);
}

extern "C" __declspec(dllexport) void SolveStereo(StereoTgv * vfs, unsigned char* fs1, unsigned char* fs2, 
	unsigned char* output, int stereoWidth, int stereoHeight, float maxDepth) {
	cv::Mat equi1, equi2;
	cv::Mat im1 = cv::Mat(cv::Size(stereoWidth, stereoHeight), CV_8UC1, fs1).clone();
	cv::Mat im2 = cv::Mat(cv::Size(stereoWidth, stereoHeight), CV_8UC1, fs2).clone();
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);

	vfs->copyImagesToDevice(equi1, equi2);
	vfs->solveStereoForwardMasked();

	cv::Mat depth8(cv::Size(stereoWidth, stereoHeight), CV_8UC1, output);
	vfs->copyStereo8ToHost(depth8, maxDepth);
}