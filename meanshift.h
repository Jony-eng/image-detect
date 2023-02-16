#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<vector>
#include<iostream>
#define PI 3.1415926535898
#define EPSILON 0.001
#define CLUSTER_EPSILON 10

using namespace cv;
using namespace std;
struct MeanShift {
	Point2f point;
	Point2f res_point;
	int label = -1;
};

void ini_meanshift(Mat src,Mat peeks,int _bandwidth);
void point_ini_meanshift(Mat src, vector<Point> vecPoint0, int _bandwidth);
void shift_once(MeanShift &p);
void label_cluster();
void DoMeanShiftCluster(vector<Point> &points);