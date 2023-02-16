#include "adtrast.h"
//--------------------
//�������ܣ���ȡͼ��ľֲ���ֵ��ֲ���׼���ͼ
//�������ƣ�getVarianceMean
//����������Mat &scr������ͼ��Ϊ��ͨ����
//����������Mat &meansDst������õ��ľ�ֵ��ͼ����ֵ��ֵ������ͼ���еĵ�һһ��Ӧ��Ϊ��ͨ����
//����������Mat &varianceDst������õ��ı�׼���ͼ����׼���ֵ������ͼ���еĵ�һһ��Ӧ��Ϊ��ͨ����
//����������int winSize���ֲ���ֵ�Ĵ��ڴ�С��ӦΪ������
//�������ͣ�bool
//--------------------

bool getVarianceMean(Mat &scr, Mat &meansDst, Mat &varianceDst, int winSize)
{
	if (!scr.data)  //�ж�ͼ���Ƿ���ȷ��ȡ��
	{
		cerr << "��ȡ�������ֵ�ĺ�������ͼƬ����";
		return false;
	}

	if (winSize % 2 == 0)
	{
		cerr << "����ֲ���ֵ���׼��Ĵ��ڴ�СӦ��Ϊ����";
		return false;
	}

	Mat copyBorder_yChannels;                        //����ͼ��߽磻
	int copyBorderSize = (winSize - 1) / 2;
	copyMakeBorder(scr, copyBorder_yChannels, copyBorderSize, copyBorderSize, copyBorderSize, copyBorderSize, BORDER_REFLECT);

	for (int i = (winSize - 1) / 2; i < copyBorder_yChannels.rows - (winSize - 1) / 2; i++)
	{
		for (int j = (winSize - 1) / 2; j < copyBorder_yChannels.cols - (winSize - 1) / 2; j++)
		{

			Mat temp = copyBorder_yChannels(Rect(j - (winSize - 1) / 2, i - (winSize - 1) / 2, winSize, winSize));   //��ȡ��չ���ͼ���е�һ�����飻

			Scalar  mean;
			Scalar  dev;
			meanStdDev(temp, mean, dev);

			varianceDst.at<float>(i - (winSize - 1) / 2, j - (winSize - 1) / 2) = dev.val[0];     ///һһ��Ӧ��ֵ��
			meansDst.at<float>(i - (winSize - 1) / 2, j - (winSize - 1) / 2) = mean.val[0];
		}
	}


	return true;
}

//--------------------------
//�������ܣ���ȡͼ��ľֲ���ֵ��ֲ���׼���ͼ
//�������ƣ�adaptContrastEnhancement
//����������Mat &scr������ͼ��Ϊ��ͨ��RGBͼ��
//����������Mat &dst����ǿ������ͼ��Ϊ��ͨ��RGBͼ��
//����������int winSize���ֲ���ֵ�Ĵ��ڴ�С��ӦΪ������
//����������int maxCg����ǿ���ȵ����ޣ�
//�������ͣ�bool
//--------------------

bool adaptContrastEnhancement(Mat &scr, Mat &dst, int winSize, int maxCg)
{
	if (!scr.data)  //�ж�ͼ���Ƿ���ȷ��ȡ��
	{
		cerr << "����Ӧ�Աȶ���ǿ��������ͼƬ����";
		return false;
	}

	Mat ycc;                        //ת���ռ䵽YCrCb��
	cvtColor(scr, ycc, COLOR_RGB2YCrCb);

	vector<Mat> channels(3);        //����ͨ����
	split(ycc, channels);


	Mat localMeansMatrix(scr.rows, scr.cols, CV_32FC1);
	Mat localVarianceMatrix(scr.rows, scr.cols, CV_32FC1);

	if (!getVarianceMean(channels[0], localMeansMatrix, localVarianceMatrix, winSize))   //��Yͨ��������ǿ��
	{
		cerr << "����ͼ���ֵ���׼������з�������";
		return false;
	}

	Mat temp = channels[0].clone();

	Scalar  mean;
	Scalar  dev;
	meanStdDev(temp, mean, dev);

	float meansGlobal = mean.val[0];
	Mat enhanceMatrix(scr.rows, scr.cols, CV_8UC1);

	for (int i = 0; i < scr.rows; i++)            //��������ÿ�����������Ӧ����
	{
		for (int j = 0; j < scr.cols; j++)
		{
			if (localVarianceMatrix.at<float>(i, j) >= 0.01)
			{
				float cg = 0.2*meansGlobal / localVarianceMatrix.at<float>(i, j);
				float cgs = cg > maxCg ? maxCg : cg;
				cgs = cgs < 1 ? 1 : cgs;

				int e = localMeansMatrix.at<float>(i, j) + cgs * (temp.at<uchar>(i, j) - localMeansMatrix.at<float>(i, j));
				if (e > 255) { e = 255; }
				else if (e < 0) { e = 0; }
				enhanceMatrix.at<uchar>(i, j) = e;
			}
			else
			{
				enhanceMatrix.at<uchar>(i, j) = temp.at<uchar>(i, j);
			}
		}

	}

	channels[0] = enhanceMatrix;    //�ϲ�ͨ����ת����ɫ�ռ�ص�RGB
	merge(channels, ycc);

	cvtColor(ycc, dst, COLOR_YCrCb2RGB);

}
