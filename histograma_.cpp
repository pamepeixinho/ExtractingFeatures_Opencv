#include <iostream>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv\cxcore.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <math.h> 
#include <vector>
#include <string.h>


using namespace cv;
using namespace std;

float result = 0;


void particiona(int *n_img, float porcao){

	result = 0;

	result = *n_img / porcao;
	*n_img = (int)(result + 0.5);

}




int main(){

	//****** HISTOGRAM (STUDY) TEST 

	vector<int> hist;

	Mat image = imread("imagem.jpg"), img_hsv, HSV;

	int h, s, v;
	float ph = 0, ps = 0, pv = 0;

	scanf("%d %d %d", &h, &s, &v);

	ph = 360.0 / (h - 1);
	ps = 255.0 / (s - 1);
	pv = 255.0 / (v - 1);


	hist.resize((h*s*v), 0);

	cvtColor(image, img_hsv, CV_BGR2HSV_FULL);
	//Convert RGB to HSV (FULL)->(0 - 360, 0-255, 0-255)

	namedWindow("original");
	namedWindow("hsv");

	imshow("original", image);
	imshow("hsv", img_hsv);

	//cvWaitKey(0);

	int k = 0;

	for (int x = 0; x < img_hsv.rows; x++){

		for (int y = 0; y < img_hsv.cols; y++)
		{


			Vec3b p = img_hsv.at<Vec3b>(x, y);

			int h1 = p.val[0];
			int s1 = p.val[1];
			int v1 = p.val[2];

			printf("H=%d, S=%d, V=%d\n", h1, s1, v1);

			particiona(&h1, ph);
			particiona(&s1, ps);
			particiona(&v1, pv);

			//	printf(" valores fc: H1=%d, S1=%d, V1=%d\n", h1, s1, v1);

			k = ((s*v)*h1) + (v*s1) + v1;
			hist[k] += 1;

			k = 0;

		}
	}

	
	printf("vetor:\n");
	for (int i = 0; i < (h*s*v); i++){
		printf("%d; ", hist[i]);
	}


	cvWaitKey(0);
	cvDestroyAllWindows();
}
