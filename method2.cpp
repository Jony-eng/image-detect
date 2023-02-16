#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

void distance_star(Mat& imge, Mat& outimge); //距离变换函数
void connected_components_stat(Mat& image);   //带统计信息
RNG rng(123);

void distance_star(Mat& imge, Mat& outimge) //距离变换函数
{
	Mat gray, binary;
	//滤波后的二值化

	//imshow("binary", binary);

	// distance transform 距离变换
	Mat dist;
	distanceTransform(imge, dist, DistanceTypes::DIST_L2, 3, CV_32F);
	normalize(dist, dist, 0, 1, NORM_MINMAX);  // 归一化函数
	imshow("距离变换图", dist);

	// binary二值化函数
	threshold(dist, outimge, 0.17, 255, THRESH_BINARY);
	imshow("距离变换结果图", outimge);
	waitKey(1000);
	return;

}



void connected_components_stat(Mat& image) {


	int sum = 0;//记录总面积
	int average = 0;//记录平均面积的1/2
	int A = 0;//豌豆数量
	int B = 0;//绿豆数量
	// 二值化
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	distance_star(binary, binary);
	Mat dist_m;
	binary.convertTo(binary, CV_8UC1);//通道转换

	//开运算、闭运算
	Mat k = getStructuringElement(MORPH_RECT, Size(13, 13), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_OPEN, k);
	//morphologyEx(binary, binary, MORPH_CLOSE, k);
	 // 形态学操作 - 彩色图像，目的是去掉干扰，让结果更好
	Mat o = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_ERODE, o);// 腐蚀，去粘连部位的干扰


	//计算连通域
	Mat labels = Mat::zeros(image.size(), CV_32S);
	Mat stats, centroids;
	int num_labels = connectedComponentsWithStats(binary, labels, stats, centroids, 8, 4);//对比新旧函数，用于过滤原始图像中轮廓分析后较小的区域，留下较大区域。

	//使用不同的颜色标记连通域
	vector<Vec3b> colors(num_labels);
	vector<Vec3b> colors1(num_labels);
	// background color
	colors[0] = Vec3b(0, 0, 0);
	colors1[0] = Vec3b(0, 0, 0);
	// object color
	for (int i = 1; i < num_labels; i++) {
		colors1[i] = Vec3b(rng.uniform(125, 125), rng.uniform(125, 125), rng.uniform(125, 125));
		colors[i] = Vec3b(rng.uniform(255, 255), rng.uniform(255, 255), rng.uniform(255, 255));
	}
	//Mat dst = Mat::zeros(image.size(), image.type());

	// render result
	Mat dst = Mat::zeros(image.size(), image.type());
	int w = image.cols;
	int h = image.rows;
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			int label = labels.at<int>(row, col);
			if (label == 0) continue;
			dst.at<Vec3b>(row, col) = colors[label];
		}
	}

	//利用统计信息标记连通域
	for (int i = 1; i < num_labels; i++) {
		Vec2d pt = centroids.at<Vec2d>(i, 0);
		int x = stats.at<int>(i, CC_STAT_LEFT);
		int y = stats.at<int>(i, CC_STAT_TOP);
		int width = stats.at<int>(i, CC_STAT_WIDTH);
		int height = stats.at<int>(i, CC_STAT_HEIGHT);
		int area = stats.at<int>(i, CC_STAT_AREA);
		sum += area;
		//printf("area : %d, center point(%.2f, %.2f)\n", area, pt[0], pt[1]);
		//circle(dst, Point(pt[0], pt[1]), 2, Scalar(0, 0, 255), -1, 8, 0);
		//rectangle(dst, Rect(x, y, width, height), Scalar(255, 0, 255), 1, 8, 0);
	}
	average = sum / num_labels;
	for (int i = 1; i < num_labels; i++)
	{
		Vec2d pt = centroids.at<Vec2d>(i, 0);
		int x = stats.at<int>(i, CC_STAT_LEFT);
		int y = stats.at<int>(i, CC_STAT_TOP);
		int width = stats.at<int>(i, CC_STAT_WIDTH);
		int height = stats.at<int>(i, CC_STAT_HEIGHT);
		int area = stats.at<int>(i, CC_STAT_AREA);
		if (area > average) {
			A = A + 1;

			circle(dst, CvPoint2D32f(pt[0], pt[1]), 2, Scalar(0, 0, 255), -1, 8, 0);
			//rectangle(dst, Rect(x, y, width, height), Scalar(255, 0, 255), 1, 8, 0);
		}
		else
		{
			B = B + 1;
			circle(dst, CvPoint2D32f(pt[0], pt[1]), 2, Scalar(255, 0, 0), -1, 8, 0);
			//rectangle(dst, Rect(x, y, width, height), Scalar(255, 255, 0), 1, 8, 0);


		}
	}
	//CString str;

	printf("豌豆 : %d, \n绿豆：%d", A, B);
	//imwrite("D://3.bmp", dst);
	imshow("二分类图像", dst);
	waitKey(0);
	//nihe(dst);椭圆拟合函数
}


void pro(Mat& img1) //处理图像，所有处理过程均通过这个函数调用处理
{

	IplImage* src = &IplImage(img1);//原彩色图像的二值化图像，用于转化通道处理

	Mat img;
	vector<Mat> color_mats;
	for (int i = 0; i < 3; i++) {
		color_mats.push_back(img1);
	}
	merge(color_mats, img);


	IplImage* dst = cvCreateImage(cvSize(src->width, src->height), 8, 3);
	CvMemStorage* stor = cvCreateMemStorage(0);
	CvSeq* cont = NULL;
	int num = cvFindContours(src, stor, &cont, sizeof(CvContour), CV_RETR_LIST);      //记录总轮廓数
	int i = 0;
	int j = 0;
	int k = 0;
	float g = 0.0;
	int z = 0;
	float w = 0.0;
	float d = 0.0;
	float flag = 0.0;//没有用到，原本想给个标签的

	float array1[700][3] = { 0.0 };//实际上没有用到这个 这个可以扩展使用




	for (i = 0; cont; cont = cont->h_next)
	{
		int* length = new   int[num];        //作用是记录每个轮廓上的点数
		length[i++] = cont->total;         //记录每个轮廓上的元素个数





		//printf("cont->total:%d\n", cont->total);
		CvPoint* point = new CvPoint[cont->total];
		CvSeqReader reader;
		CvPoint pt = cvPoint(0, 0);
		CvPoint pt1 = cvPoint(0, 0);
		CvPoint pt2 = cvPoint(0, 0);
		CvPoint pt3 = cvPoint(0, 0);
		CvPoint pt4 = cvPoint(0, 0);
		cvStartReadSeq(cont, &reader);
		for (int j = 0; j < cont->total; j++)
		{
			CV_READ_SEQ_ELEM(pt, reader);
			point[j] = pt;
			//cout << pt.x << "  " << pt.y << endl;
		}
		for (j = 0; j < cont->total; j++)
		{
			int k = (j + 1) % cont->total;
			cvLine(dst, point[j], point[k], cvScalar(0, 0, 255), 1, 4, 0);
		}


		for (int j = 0; j < cont->total - 14; j++)
		{
			CV_READ_SEQ_ELEM(pt1, reader);
			CV_READ_SEQ_ELEM(pt2, reader);
			CV_READ_SEQ_ELEM(pt3, reader);
			pt1 = point[j];
			pt2 = point[j + 7];    //取步长为7
			pt3 = point[j + 14];


			if ((pt1.x != pt2.x) & (pt3.x != pt2.x))
			{
				//w = j + 8;
				//int g = int(((pt1.y-pt2.y)/(pt1.x-pt2.x))+ ((pt2.y - pt3.y) / (pt2.x - pt3.x)));
				float g = floor((pt2.x - pt1.x) * (pt3.x - pt2.x) + (pt2.y - pt1.y) * (pt3.y - pt2.y));
				if (g < 1.0)
				{
					//w = j + 8;
					flag = 1;

					//cvLine(dst, point[j], point[j+2], cvScalar(0, 255, 0), 1, 4, 0);
					cvCircle(dst, pt2, 1, Scalar(0, 255, 255), -1);
					array1[z][0] = floor(pt2.x);
					array1[z][1] = floor(pt2.y);
					array1[z][2] = flag;
					z++;

				}
			}
		}

		delete point;
	}


	for (i = 0; i < 700; i++)
	{
		if (array1[i][2] == 1.0)
		{
			for (j = 1; j < 700; j++) {
				w = 35.0;
				float d = sqrt(pow((array1[i][0] - array1[j][0]), 2) +
					pow((array1[i][1] - array1[j][1]), 2));
				if ((d < w) & (d > 5.0)) {
					//w = d;
					CvPoint2D32f pt5 = cvPoint2D32f(0.0, 0.0);
					CvPoint2D32f pt6 = cvPoint2D32f(0.0, 0.0);
					//CvPoint pt6 = cvPoint(0, 0);
					pt5.x = array1[i][0];
					pt5.y = array1[i][1];
					pt6.x = array1[j][0];
					pt6.y = array1[j][1];

					//凹点分割画线
					line(img, pt5, pt6, Scalar(0, 0, 0), 2, 4, 0);
				}
			}
		}
		else {

		}

	}
	cvShowImage("寻找凹点图", dst);

	imshow("凹点分割后", img);
	waitKey(1000);
	connected_components_stat(img);
	return;
}
