#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>

#include <stereotgv/stereotgv.h>

void showDepthJet(std::string windowName, cv::Mat image, float maxDepth, bool shouldWait = true) {
	cv::Mat u_norm, u_gray, u_color;
	u_norm = image * 256.0f / maxDepth;
	u_norm.convertTo(u_gray, CV_8UC1);
	cv::applyColorMap(u_gray, u_color, cv::COLORMAP_JET);

	cv::imshow(windowName, u_color);
	if (shouldWait) cv::waitKey();
}

// For using the dataset on the github page
int test_equidistant_data() {
	std::string folder = "C:/Users/menandro/Desktop/Conferences/ICRA2020/icra_dataset/";
	std::string filename = "im203";
	std::string outputFilename = folder + "output/" + filename + ".flo";
	cv::Mat im1 = cv::imread(folder + "image_02/data/" + filename + ".png");
	cv::Mat im2 = cv::imread(folder + "image_03/data/" + filename + ".png");

	// VFS Parameters
	StereoTgv* stereotgv = new StereoTgv();
	int width = 800;
	int height = 800;
	float stereoScaling = 1.0f;
	int nLevel = 11;
	float fScale = 1.2f;
	int nWarpIters = 50;
	int nSolverIters = 50;
	float lambda = 5.0f;
	stereotgv->limitRange = 0.1f;
	float beta = 9.0f;//4.0f;
	float gamma = 0.85f;// 0.2f;
	float alpha0 = 17.0f;// 5.0f;
	float alpha1 = 1.2f;// 1.0f;
	float timeStepLambda = 1.0f;

	stereotgv->initialize(width, height, beta, gamma, alpha0, alpha1,
		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
	stereotgv->visualizeResults = true;

	cv::Mat mask = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	circle(mask, cv::Point(width / 2, height / 2), width / 2 - 10, cv::Scalar(256.0f), -1);
	cv::Mat fisheyeMask;
	mask.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
	stereotgv->copyMaskToDevice(fisheyeMask);
	
	cv::Mat equi1, equi2;
	cv::cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(im2, im2, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(im1, equi1);
	cv::equalizeHist(im2, equi2);
	int stereoWidth = width;
	int stereoHeight = height;
	cv::Mat translationVector = cv::readOpticalFlow(folder + "translationVector/" + filename + ".flo");
	cv::Mat calibrationVector = cv::readOpticalFlow(folder + "calibrationVector/" + filename + ".flo");

	stereotgv->loadVectorFields(translationVector, calibrationVector);

	clock_t start = clock();

	stereotgv->copyImagesToDevice(equi1, equi2);
	stereotgv->solveStereoForwardMasked();

	cv::Mat disparityVis = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
	stereotgv->copyDisparityVisToHost(disparityVis, 50.0f);
	cv::imshow("flow", disparityVis);

	clock_t timeElapsed = (clock() - start);
	std::cout << "ours time: " << timeElapsed << " ms" << std::endl;

	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
	stereotgv->copyDisparityToHost(disparity);
	cv::writeOpticalFlow(outputFilename, disparity);

	cv::waitKey();
}

// For using Realsense T265
int test_t265_data() {
	cv::Mat im1 = cv::imread("robot1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat im2 = cv::imread("robot2.png", cv::IMREAD_GRAYSCALE);
	if (im1.empty()) {
		std::cout << "Error: im1 not found." << std::endl;
		return -1;
	}
	if (im2.empty()) {
		std::cout << "Error: im1 not found." << std::endl;
		return -1;
	}

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
		if (translationVector.empty()) {
			std::cout << "Error: translationVector not found." << std::endl;
			return -1;
		}
		if (calibrationVector.empty()) {
			std::cout << "Error: calibrationVector not found." << std::endl;
			return -1;
		}
		if (fisheyeMask8.empty()) {
			std::cout << "Error: fisheyeMask8 not found." << std::endl;
			return -1;
		}
	}
	else {
		translationVector = cv::readOpticalFlow("translationVectorHalf.flo");
		calibrationVector = cv::readOpticalFlow("calibrationVectorHalf.flo");
		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
		if (translationVector.empty()) {
			std::cout << "Error: translationVector not found." << std::endl;
			return -1;
		}
		if (calibrationVector.empty()) {
			std::cout << "Error: calibrationVector not found." << std::endl;
			return -1;
		}
		if (fisheyeMask8.empty()) {
			std::cout << "Error: fisheyeMask8 not found." << std::endl;
			return -1;
		}
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

//int test_t265_data_wang() {
//	cv::Mat im1 = cv::imread("wang1.png", cv::IMREAD_GRAYSCALE);
//	cv::Mat im2 = cv::imread("wang2.png", cv::IMREAD_GRAYSCALE);
//
//	// Camera parameters
//	float focalx = 286.518f;
//	float focaly = 286.602f;
//	float cx = 423.142f;
//	float cy = 393.321f;
//	// Kanala-Brandt distortion paramters (from t265)
//	float d0 = -0.00727452f;
//	float d1 = 0.05200624f;
//	float d2 = -0.05105740f;
//	float d3 = 0.01108238f;
//	// Camera translation
//	float tx = -0.0639505f;
//	float ty = 0.000184912f;
//	float tz = -0.0000766807f;
//
//	// VFS parameters
//	StereoTgv* stereotgv = new StereoTgv();
//	int width = 848;
//	int height = 800;
//	/*
//	float stereoScaling = 1.0f; // 1.0 and 2.0 only
//	int nLevel = 11;		// Limith such that minimum width > 50pix
//	float fScale = 1.2f;	// Ideally 2.0
//	int nWarpIters = 100;	// Change to reduce processing time
//	int nSolverIters = 100;	// Change to reduce processing time
//	float lambda = 5.0f;	// Change to increase data dependency
//	float beta = 9.0f;
//	float gamma = 0.85f;
//	float alpha0 = 17.0f;
//	float alpha1 = 1.2f;
//	float timeStepLambda = 1.0f;
//	stereotgv->limitRange = 0.1f;
//	*/
//	float stereoScaling = 2.0f; // 1.0 and 2.0 only
//	int nLevel = 5;		// Limith such that minimum width > 50pix
//	float fScale = 2.0f;	// Ideally 2.0
//	int nWarpIters = 20;	// Change to reduce processing time
//	int nSolverIters = 10;	// Change to reduce processing time
//	float lambda = 5.0f;	// Change to increase data dependency
//	float beta = 9.0f;
//	float gamma = 0.85f;
//	float alpha0 = 17.0f;
//	float alpha1 = 1.2f;
//	float timeStepLambda = 1.0f;
//	stereotgv->limitRange = 0.1f;
//
//	int stereoWidth = (int)((float)width / stereoScaling);
//	int stereoHeight = (int)((float)height / stereoScaling);
//	stereotgv->initialize(stereoWidth, stereoHeight, beta, gamma, alpha0, alpha1,
//		timeStepLambda, lambda, nLevel, fScale, nWarpIters, nSolverIters);
//	stereotgv->visualizeResults = true;
//
//	// Load fisheye mask and vector fields
//	cv::Mat translationVector, calibrationVector, fisheyeMask8, fisheyeMask;
//	if (stereoScaling == 1.0f) {
//		translationVector = cv::readOpticalFlow("translationVectorWang.flo");
//		calibrationVector = cv::readOpticalFlow("calibrationVectorWang.flo");
//		fisheyeMask8 = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
//	}
//	else {
//		translationVector = cv::readOpticalFlow("translationVectorWangHalf.flo");
//		calibrationVector = cv::readOpticalFlow("calibrationVectorWangHalf.flo");
//		fisheyeMask8 = cv::imread("maskHalf.png", cv::IMREAD_GRAYSCALE);
//	}
//	fisheyeMask8.convertTo(fisheyeMask, CV_32F, 1.0 / 255.0);
//	stereotgv->copyMaskToDevice(fisheyeMask);
//	stereotgv->loadVectorFields(translationVector, calibrationVector);
//
//	// Solve stereo depth
//	cv::Mat equi1, equi2;
//	cv::equalizeHist(im1, equi1);
//	cv::equalizeHist(im2, equi2);
//	cv::resize(equi1, equi1, cv::Size(stereoWidth, stereoHeight));
//	cv::resize(equi2, equi2, cv::Size(stereoWidth, stereoHeight));
//	cv::imshow("input", equi1);
//
//	clock_t start = clock();
//	stereotgv->copyImagesToDevice(equi1, equi2);
//	stereotgv->solveStereoForwardMasked();
//
//	cv::Mat depth = cv::Mat(stereoHeight, stereoWidth, CV_32F);
//	cv::Mat X = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
//	stereotgv->copyStereoToHost(depth, X, focalx / stereoScaling, focaly / stereoScaling,
//		cx / stereoScaling, cy / stereoScaling,
//		d0, d1, d2, d3,
//		tx, ty, tz);
//	clock_t timeElapsed = (clock() - start);
//	cv::imshow("X", X);
//	cv::Mat xyz[3];
//	cv::split(X, xyz);
//	cv::Mat Z16, depth16;
//	xyz[2].convertTo(Z16, CV_16U, 256.0);
//	depth.convertTo(depth16, CV_16U, 256.0);
//	cv::imwrite("z16.png", Z16);
//	cv::imwrite("depth16.png", depth16);
//	/*std::cout << xyz[2].at<float>(495, 243) << " " << xyz[2].at<float>(454, 469) << " " << xyz[2].at<float>(527, 669)
//		<< " " << xyz[2].at<float>(194, 437) << std::endl;*/
//
//	cv::Mat disparity = cv::Mat(stereoHeight, stereoWidth, CV_32FC2);
//	stereotgv->copyDisparityToHost(disparity);
//	std::cout << "time: " << timeElapsed << " ms ";
//	cv::writeOpticalFlow("disparity.flo", disparity);
//
//	cv::Mat uvrgb = cv::Mat(stereoHeight, stereoWidth, CV_32FC3);
//	stereotgv->copyDisparityVisToHost(uvrgb, 40.0f);
//	cv::Mat uvrgb8;
//	uvrgb.convertTo(uvrgb8, CV_8UC3, 255.0);
//	cv::imshow("2D disparity", uvrgb8);
//	cv::waitKey();
//	return 0;
//}

int main() {
	// Choose which to run
	//return test_equidistant_data();
	return test_t265_data();
	//return test_t265_data_wang();
}