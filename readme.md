# Variational Fisheye Stereo

Implementation of our paper submission for ICRA2020.

Menandro Roxas and Takeshi Oishi, "Real-Time Variational Fisheye Stereo without Rectification and Undistortion," arXiv preprint 

## Requirements

1. OpenCV, OpenCV Contrib (optflow) (tested with v4.10.0)
2. CUDA 10.1 (Including Samples for headers)
3. Visual Studio 2019

## Building Instructions
The solution consists of two projects - stereotgv and test_vfs. stereotgv generates a static library from which test_vfs links. test_vfs generates a Win32 .exe file. 

There is a lib_link.h header (for both project) that links the necessary libraries. Modify the directories:

```
#define LIB_PATH "D:/dev/lib64/"
#define CV_LIB_PATH "D:/dev/lib64/"
#define CUDA_LIB_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/"
```

to point to the location of OpenCV (CV_LIB_PATH) and CUDA (CUDA_LIB_PATH) .lib files.

At the same time, modify the Project Properties -> VC++ Directories -> (Executables, Includes, and Libraries) to point to the location of the OpenCV and CUDA builds, too.

```
Executable: D:/dev/bin
Includes: D:/dev/include
Libraries: D:/dev/lib64
```

## Results
INTEL Realsense T265
[Video 01](http://b2.cvl.iis.u-tokyo.ac.jp/~roxas/output001.mp4)
[Video 04](http://b2.cvl.iis.u-tokyo.ac.jp/~roxas/output004.mp4)
[Video 07](http://b2.cvl.iis.u-tokyo.ac.jp/~roxas/output007.mp4)
[Video 16](http://b2.cvl.iis.u-tokyo.ac.jp/~roxas/output016.mp4)
[Video 19](http://b2.cvl.iis.u-tokyo.ac.jp/~roxas/output019.mp4)
[Video 20](http://b2.cvl.iis.u-tokyo.ac.jp/~roxas/output020.mp4)
[Video 21](http://b2.cvl.iis.u-tokyo.ac.jp/~roxas/outputbuggy.mp4)

### To do
*CMake

## License
This project is licensed under the MIT license

## Author
Menandro Roxas, 3D Vision Laboratory (Oishi Laboratory), Institute of Industrial Science, The University of Tokyo


