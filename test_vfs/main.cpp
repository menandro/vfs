#include <opencv2/opencv.hpp>
#include <stereotgv/stereotgv.h>

#if _WIN64
#define LIB_PATH "D:/dev/lib64/"
#define CV_LIB_PATH "D:/dev/lib64/"
#else
#define LIB_PATH "D:/dev/staticlib32/"
#endif

#ifdef _DEBUG
#define LIB_EXT "d.lib"
#else
#define LIB_EXT ".lib"
#endif

#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/"
#pragma comment(lib, CUDA_LIB_PATH "cudart.lib")

#define CV_VER_NUM CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, LIB_PATH "opencv_core" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_highgui" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_videoio" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_imgproc" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_calib3d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_xfeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_optflow" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_imgcodecs" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_features2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_tracking" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_flann" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_cudafeatures2d" CV_VER_NUM LIB_EXT)
#pragma comment(lib, LIB_PATH "opencv_cudaimgproc" CV_VER_NUM LIB_EXT)

#pragma comment(lib, LIB_PATH "opencv_video" CV_VER_NUM LIB_EXT)
void showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait = true) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

int main() {
	cv::Mat im1 = cv::imread("fs1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("fs2.png", cv::IMREAD_GRAYSCALE);

	StereoTgv* stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f; // Bugged, don't change
	int nLevel = 4;		// Limith such that minimum width > 50pix
	float fScale = 2.0f;	// Ideally 2.0
	int nWarpIters = 5;	// Change to reduce processing time
	int nSolverIters = 5;	// Change to reduce processing time
	float lambda = 5.0f;	// Change to increase data dependency
	float beta = 9.0f;		
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	//stereotgv->baseline = 0.0642f; // Not used
	//stereotgv->focal = 285.8557f / stereoScaling;
	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	// Load fisheye Mask
	cv::Mat fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	cv::Mat fisheyeMask;
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);

	// Load vector fields
	cv::Mat translationVector, calibrationVector;
	translationVector = cv::readOpticalFlow("translationVector.flo");
	calibrationVector = cv::readOpticalFlow("calibrationVector.flo");
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	// Solve stereo depth
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);

	clock_t start = clock();
	stereotgv->copyImagesToDevice(equi1, equi2);
	stereotgv->solveStereoForwardMasked();

	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
	stereotgv->copyDisparityToHost(disparity);
	clock_t timeElapsed = (clock() - start);
	std::cout << "time: " << timeElapsed << " ms ";
	cv::writeOpticalFlow("disparity.flo", disparity);
	cv::waitKey();
	return 0;
}