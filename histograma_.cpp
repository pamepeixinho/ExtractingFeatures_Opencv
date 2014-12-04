#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv\cxcore.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <math.h> 
#include <vector>
#include <string.h>

using namespace std;
using namespace cv;

vector<float> normalizavetor(vector<float>hist, int t, int size){
	t = (float)t;
	int i = 0;
	for (i = 0; i < size; i++)
		hist[i] = hist[i]/t;
	return hist;
}

void HistogramaMask(char *nomearquivo, Mat image, Mat mask, int h, int s, int v){

	FILE* arq = fopen(nomearquivo, "w");

	vector<float> hist;

	Mat img_hsv, HSV;

	float ph = 0, ps = 0, pv = 0;

	ph = 360.0 / (h - 1);
	ps = 255.0 / (s - 1);
	pv = 255.0 / (v - 1);


	hist.resize((h*s*v), 0);

	cvtColor(image, img_hsv, CV_BGR2HSV_FULL);
		//Convert RGB to HSV (FULL)->(0 - 360, 0-255, 0-255)

	imshow("original", image);
	imshow("hsv", img_hsv);
	imshow("mask", mask);


	int k = 0, branco = 0, h1 = 0, s1 = 0, v1 = 0, r=0, g=0, b=0;

	for (int x = 0; x < img_hsv.rows; x++){

		for (int y = 0; y < img_hsv.cols; y++){

			Vec3b m = mask.at<Vec3b>(x, y);

			r = m.val[0];
			g = m.val[1];
			b = m.val[2];

			fprintf(arq, "\n\nimage mascara = %d %d %d\n", r,b,g);

			Vec3b p = img_hsv.at<Vec3b>(x, y);

			h1 = p.val[0];
			s1 = p.val[1];
			v1 = p.val[2];

			//printf("H=%d, S=%d, V=%d\n", h1, s1, v1);
			fprintf(arq, "H=%d, S=%d, V=%d\n", h1, s1, v1);

			if (r!=0 && g!=0 && b!=0){

				//printf("H(entravetor)=%d, S(entravetor)=%d, V(entravetor)=%d\n", h1, s1, v1);
				fprintf(arq, "H(entravetor)=%d, S(entravetor)=%d, V(entravetor)=%d\n", h1, s1, v1);

				particiona(&h1, ph);
				particiona(&s1, ps);
				particiona(&v1, pv);

				k = ((s*v)*h1) + (v*s1) + v1;
				hist[k] += 1;

				k = 0;
				branco++;
			}

		}
	}

	//printf("%d\n", t);
	fprintf(arq, "quantidade pixel que entrou no vetor:%d\n", branco);
	fprintf(arq, "\nImage:quantidade pixel linhas:%d\ colunas:%d e total:%d\n", img_hsv.rows, img_hsv.cols, (img_hsv.rows*img_hsv.cols));
	fprintf(arq, "\nMask:quantidade pixel linhas:%d\ colunas:%d e total:%d\n", mask.rows, mask.cols, (mask.rows*mask.cols));

	//printf("vetor:\n");
	fprintf(arq, "\nVetor antes: \n");

	for (int i = 0; i < (h*s*v); i++){
		//	printf("%d ", hist[i]);
		fprintf(arq, "%.1f ", hist[i]);
	}
	
	hist = normalizavetor(hist, branco, (h*s*v));
	fprintf(arq, "\n\nVetor depois: \n");
	for (int i = 0; i < (h*s*v); i++){
		//	printf("%d ", hist[i]);
		fprintf(arq, "%.2f ", hist[i]);
	}

	fclose(arq);
}

Mat ApplymaskHist(Mat image, Mat mask){
	Mat final = Mat::zeros(3, 3, CV_8UC1);
	
	image.copyTo(final, mask);


	imshow("final", final);
	imshow("imagem", image);
	imshow("mask", mask);

	return final;
}



int main(int argc, char *argv[]){

	char *nomeimage = argv[1], *nomemask = argv[2],*nomearquivo = argv[3];
	int arg_h = atoi(argv[4]), arg_s = atoi(argv[5]), arg_v = atoi(argv[6]);

	Mat image = imread(nomeimage), mask = imread(nomemask), final;

	HistogramaMask(nomearquivo, image, mask, arg_h, arg_s, arg_v);
	
	cvWaitKey(0);
	cvDestroyAllWindows();

	return 0;
}
