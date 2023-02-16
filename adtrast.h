#pragma once
#include<opencv2/opencv.hpp>
#include<vector>
#include<iostream>

using namespace std;
using namespace cv;

bool getVarianceMean(Mat &scr, Mat &meansDst, Mat &varianceDst, int winSize);
bool adaptContrastEnhancement(Mat &scr, Mat &dst, int winSize, int maxCg);
