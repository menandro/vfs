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

#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/lib/x64/"
#pragma comment(lib, CUDA_LIB_PATH "cudart.lib")

#define CV_VER_NUM CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, CV_LIB_PATH "opencv_core" CV_VER_NUM LIB_EXT)
#pragma comment(lib, CV_LIB_PATH "opencv_highgui" CV_VER_NUM LIB_EXT) // imshow
#pragma comment(lib, LIB_PATH "opencv_imgproc" CV_VER_NUM LIB_EXT) // Equilize Histrogram
#pragma comment(lib, LIB_PATH "opencv_imgcodecs" CV_VER_NUM LIB_EXT) // Opening PNG
#pragma comment(lib, LIB_PATH "opencv_video" CV_VER_NUM LIB_EXT) // Write Optical Flow

void showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait = true) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

int main() {
	cv::Mat im1 = cv::imread("robot1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("robot2.png", cv::IMREAD_GRAYSCALE);

	// Camera parameters
	float focalx = 285.722f;
	float focaly = 286.759f;
	float cx = 420.135f;
	float cy = 403.394f;
	// Kanala-Brandt distortion paramters (from t265)
	float d0 = -0.00659769f;
	float d1 = 0.0473251f;
	float d2 = -0.0458264f;
	float d3 = 0.00897725f;
	// Camera translation
	float tx = -0.0641854f;
	float ty = -0.000218299f;
	float tz = 0.000111253f;

	// VFS parameters
	StereoTgv* stereotgv = new StereoTgv();
	int width = 848;
	int height = 800;
	float stereoScaling = 1.0f; // 1.0 and 2.0 only
	int nLevel = 11;		// Limith such that minimum width > 50pix
	float fScale = 1.2f;	// Ideally 2.0
	int nWarpIters = 100;	// Change to reduce processing time
	int nSolverIters = 100;	// Change to reduce processing time
	float lambda = 5.0f;	// Change to increase data dependency
	float beta = 9.0f;
	float gamma = 0.85f;
	float alpha0 = 17.0f;
	float alpha1 = 1.2f;
	float timeStepLambda = 1.0f;
	stereotgv->limitRange = 0.1f;

	int stereoWidth = (int)((float)width / stereoScaling);
	int stereoHeight = (int)((float)height / stereoScaling);
	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	// Load fisheye mask and vector fields
	cv::Mat translationVector, calibrationVector, fisheyeMask8, fisheyeMask;
	if (stereoScaling == 1.0f) {
		translationVector = cv::readOpticalFlow("translationVector.flo");
		calibrationVector = cv::readOpticalFlow("calibrationVector.flo");
		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
	}
	else {
		translationVector = cv::readOpticalFlow("translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("calibrationVectorHalf.flo");
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
	}
	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);
	stereotgv->loadVectorFields(translationVector, calibrationVector);

	// Solve stereo depth
	cv::Mat equi1, equi2;
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	cv::resize(equi1, equi1, cv::Size(stereoWidth, stereoHeight));
	cv::resize(equi2, equi2, cv::Size(stereoWidth, stereoHeight));
	cv::imshow("input", equi1);

	clock_t start = clock();
	stereotgv->copyImagesToDevice(equi1, equi2);
	stereotgv->solveStereoForwardMasked();

	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
	cv::Mat X = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	stereotgv->copyStereoToHost(depth, X, focalx / stereoScaling, focaly / stereoScaling,
		cx / stereoScaling, cy / stereoScaling,
		d0, d1, d2, d3,
		tx, ty, tz);
	clock_t timeElapsed = (clock() - start);
	cv::imshow("X", X);

	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
	stereotgv->copyDisparityToHost(disparity);
	std::cout << "time: " << timeElapsed << " ms ";
	cv::writeOpticalFlow("disparity.flo", disparity);

	cv::Mat uvrgb = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	stereotgv->copyDisparityVisToHost(uvrgb, 40.0f);
	cv::Mat uvrgb8;
	uvrgb.convertTo(uvrgb8, CV_8UC3, 255.0);
	cv::imshow("2D disparity", uvrgb8);
	cv::waitKey();
	return 0;
}