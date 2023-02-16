#include "meanshift.h"
bool stop;
int point_num;
int cluster_num;
int kernel_bandwidth;
vector<MeanShift> dataset;
int GetManhattanDistance(Point2f p0, Point2f p1)
{
	return abs(p0.x - p1.x) + abs(p0.y - p1.y);
}

float GetEuclideanDistance(Point2f p0, Point2f p1)
{
	return sqrt((p0.x - p1.x)*(p0.x - p1.x) + (p0.y - p1.y)*(p0.y - p1.y));
}

float GaussianKernel(int distance, int bandwidth)
{
	return exp(-0.5*(distance*distance) / (bandwidth*bandwidth));
}
void ini_meanshift(Mat src,Mat peeks,int _bandwidth) {
	//初始化dataset
	for (int x = 0; x < peeks.cols; x++)
	{
		for (int y = 0; y < peeks.rows; y++)
		{
			Point pos;
			pos.x = x;
			pos.y = y;
			if (peeks.at<uchar>(y, x) && src.at<uchar>(y, x) > 70) {
				MeanShift tmp;
				tmp.point.x = x;
				tmp.point.y = y;
				tmp.res_point.x = x;
				tmp.res_point.y = y;
				dataset.push_back(tmp);
			}			
		}
	}
	//初始化其他参数
	point_num = dataset.size();
	kernel_bandwidth = _bandwidth;
	stop = false;
}
void point_ini_meanshift(Mat src, vector<Point> vecPoint0, int _bandwidth) {
	//初始化dataset
	for (int i = 0; i < vecPoint0.size(); i++) {
		int y = vecPoint0[i].y;
		int x = vecPoint0[i].x;
		if (src.at<uchar>(y, x) > 70) {
			MeanShift tmp;
			tmp.point.x = x;
			tmp.point.y = y;
			tmp.res_point.x = x;
			tmp.res_point.y = y;
			dataset.push_back(tmp);
		}
	}
	//初始化其他参数
	point_num = dataset.size();
	kernel_bandwidth = _bandwidth;
	stop = false;
}
void shift_once(MeanShift &p) {
	float x_sum = 0;
	float y_sum = 0;
	float weight_sum = 0;
	for (int i = 0; i < point_num; i++)
	{
		float tmp_distance = GetEuclideanDistance(p.res_point, dataset[i].point);
		float weight = GaussianKernel(tmp_distance, kernel_bandwidth);
		x_sum += dataset[i].point.x * weight;
		y_sum += dataset[i].point.y * weight;
		weight_sum += weight;
	}
	Point2f shift_vector(x_sum / weight_sum, y_sum / weight_sum);
	float shift_distance = GetEuclideanDistance(p.res_point, shift_vector);
	//cout << "shift_distance = " << shift_distance << endl;

	if (shift_distance < EPSILON)
		stop = true;

	p.res_point = shift_vector;
}
void label_cluster() {
	int current_label = -1;
	for (int i = 0; i < point_num; i++)
	{
		if (dataset[i].label != -1)
			continue;
		current_label++;
		for (int j = 0; j < point_num; j++)
		{
			if (GetEuclideanDistance(dataset[i].res_point, dataset[j].res_point) < CLUSTER_EPSILON)
			{
				dataset[j].label = current_label;
			}
		}
	}
}
void DoMeanShiftCluster(vector<Point> &points)
{
	for (int i = 0; i < point_num; i++)
	{
		stop = false;
		while (!stop)
		{
			shift_once(dataset[i]);
		}
		//cout << "(" << dataset[i].res_point.x << "," << dataset[i].res_point.y << ")" << endl;
	}
	label_cluster();
	Point result_point;
	for (int i = 0; i < dataset.size(); i++)
	{
		result_point.x = dataset[i].res_point.x;
		result_point.y = dataset[i].res_point.y;
		points.push_back(result_point);
	}
}
