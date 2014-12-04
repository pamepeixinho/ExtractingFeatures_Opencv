//PAMELA PEIXINHO_ OPENCV FEATURES
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


	//TRACKBAR
Mat image = imread("imagem.jpg"), img;
	//trackbar - blur
int slider = 5;
float sigma = 0.3*((slider - 1)*0.5 - 1) + 0.8;

void on_trackbar(int, void *){
	int k_size = max(1, slider);
	k_size = (k_size % 2 == 0) ? k_size + 1 : k_size;
	sigma = 0.3*((k_size - 1)*0.5 - 1) + 0.8;
	GaussianBlur(image, img, Size(k_size, k_size), sigma);
	imshow("blurred img", img);
}

	//FUNCTION to HISTOGRAMA
float result = 0;
void particiona(int *n_img, float porcao){
	
	result = 0;

	result = *n_img / porcao;
	*n_img = (int)(result + 0.5);

}

void fouriertransexample(){
	Mat I = imread("image.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).

	imshow("Input Image", I);    // Show the result
	imshow("spectrum magnitude", magI);

}

void ContornoHullMAT(){

	Mat image, imagegray, imagethresh, contourout, imagemask = Mat::zeros(3,3,CV_8UC1);

	image = imread("maos.jpg");

	cvtColor(image, imagegray, CV_BGR2GRAY);
	GaussianBlur(imagegray, imagegray, Size(3,3), 4.1);
	GaussianBlur(imagegray, imagegray, Size(3, 3), 3.1);

	threshold(imagegray, imagethresh, 128, 255, CV_THRESH_BINARY);

	image.copyTo(imagemask, imagethresh);

	vector<vector <Point> > contours;

	contourout = imagethresh.clone();

	//find contour
	findContours(contourout, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(contours.size());
	vector<vector<Point> >contouraprx(contours.size());

	for (int i = 0; i < contours.size(); i++)
		convexHull(Mat(contours[i]), hull[i], false);

	for (size_t k = 0; k < contours.size(); k++)
		approxPolyDP(Mat(contours[k]), contouraprx[k], 5, true);

	//Images contour to draw
	Mat contourimage(image.size(), CV_8UC3, Scalar(0, 0, 0));
	Mat hullimage(image.size(), CV_8UC3, Scalar(0, 0, 0));
	Mat hullonly(image.size(), CV_8UC3, Scalar(0, 0, 0));
	Mat approximateimage(image.size(), CV_8UC3, Scalar(0, 0, 0));
	Mat imagefourier(image.size(), CV_8UC3, Scalar(0, 0, 0));


	//doing it colorful
	Scalar colors[3];
	colors[0] = Scalar(0, 0, 255);
	colors[1] = Scalar(0, 255, 0);
	colors[2] = Scalar(0, 0, 255);

	for (size_t id = 0; id < contours.size(); id++){
		drawContours(contourimage, contours, id, colors[id % 3]);
		drawContours(hullimage, hull, id, colors[id % 3]);
		drawContours(hullimage, contours, id, colors[id % 3]);
		drawContours(approximateimage, contouraprx, id, colors[id % 3]);
		drawContours(hullonly, hull, id, colors[id % 3]);
		

	}
		
	int x = image.rows;
	int y = image.cols;

	//printf("%d %d", x, y);

	//dft(contourimage, imagefourier, DFT_SCALE, contourimage.rows);

	imshow("input", image);
	cvMoveWindow("input", 0, 0);

	imshow("gray", imagegray);
	cvMoveWindow("gray", x, 0);

	imshow("thresh", imagethresh);
	cvMoveWindow("thresh", 0, y);

	imshow("contour", contourimage);
	cvMoveWindow("contour", x, y);

	imshow("hullplus", hullimage);
	cvMoveWindow("hullplus", (2*x), 0);
	
	imshow("Approx", approximateimage);
	cvMoveWindow("Approx", (2*x), y);
	
	imshow("hull", hullonly);
	cvMoveWindow("hull", (3 * x), 0);

	imshow("mask", imagemask);
	cvMoveWindow("maks", (3 * x), y);
	

}

void VideoHSVInverse(){
	
	int c = 0;
	CvCapture* capture = cvCaptureFromCAM(0);

	if (!cvQueryFrame(capture)){ cout << "Video capture failed, please check the camera." << endl; }
	else{ cout << "Video camera capture status: OK" << endl; };

	CvSize sz = cvGetSize(cvQueryFrame(capture));

	IplImage* src = cvCreateImage(sz, 8, 3);

	IplImage* hsv_image = cvCreateImage(sz, 8, 3);

	IplImage* hsv_mask = cvCreateImage(sz, 8, 1);

	CvScalar  hsv_min = cvScalar(0, 30, 80, 0);

	CvScalar  hsv_max = cvScalar(20, 150, 255, 0);

	while (c != 27)	{

		src = cvQueryFrame(capture);

		cvNamedWindow("src", 1); cvShowImage("src", src);

		cvCvtColor(src, hsv_image, CV_BGR2HSV);

		cvNamedWindow("hsv-img", 1); cvShowImage("hsv-img", hsv_image);

		cvInRangeS(hsv_image, hsv_min, hsv_max, hsv_mask);

		cvNamedWindow("hsv-msk", 1); cvShowImage("hsv-msk", hsv_mask); hsv_mask->origin = 1;

		c = cvWaitKey(10);

	}

	cvReleaseCapture(&capture);

}

void ContornoIplImage(){

	IplImage *image = cvLoadImage("Maos.jpg", CV_LOAD_IMAGE_GRAYSCALE), *img1 = 0, *img2 = 0, *img, *img3, *img4, *img5;

	img = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
	img1 = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
	img2 = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
	img3 = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
	img4 = cvLoadImage("imagem.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	img5 = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);

	//cvSmooth(image, img, CV_BLUR, 3);

	//canny
	cvCanny(image, img1, 10, 30, 3);

	//	threshold
	//	cvAdaptiveThreshold(image, img2, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C,  CV_THRESH_BINARY, 3, 5.0);
	cvThreshold(image, img2, 100, 255, CV_THRESH_BINARY);

	cvNamedWindow("canny", 1);
	cvNamedWindow("adptative Threshold", 1);

	cvShowImage("canny", img1);
	cvShowImage("adptative Threshold", img2);

	cvNamedWindow("original", 1);
	cvShowImage("original", image);

	CvMemStorage *storage = cvCreateMemStorage(), *storage_poly = cvCreateMemStorage();
	CvSeq *contours = 0, *hull = 0, *poly;

	int nc = 0, nf = 0;
	//nc = cvFindContours(img1, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	nf = cvFindContours(img2, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	//Numero de contornos canny: %d\n
	printf("Numero de contornos threshold: %d\n", nf);

	//hull = cvConvexHull2(contours, storage, CV_CLOCKWISE, 0);

	poly = cvApproxPoly(contours, sizeof(CvContour), storage_poly, CV_POLY_APPROX_DP, 0);


	cvZero(img3);
	//cvZero(img5);

	cvDrawContours(img3, contours, cvScalarAll(255), cvScalarAll(255), 100);
	cvDrawContours(img4, contours, cvScalarAll(255), cvScalarAll(255), 100);
	cvDrawContours(img5, poly, cvScalarAll(255), cvScalarAll(255), 100);
	//cvDrawContours(img5, hull, cvScalarAll(255), cvScalarAll(255), 100);


	cvNamedWindow("contourn", 1);
	cvShowImage("contourn", img3);

	//cvNamedWindow("contourn image", 1);
	//cvShowImage("contourn image", img4);

	/*cvNamedWindow("hull", 1);
	cvShowImage("hull", img5);*/

	cvNamedWindow("POLY APPROXIMATE", 3);
	cvShowImage("POLY APPROXIMATE", img5);

}
void CalcHistograma(){
	Mat src,img;

	/// Load image
	src = imread("image.jpg");
	cvtColor(src, img, CV_BGR2HSV);

	/*
	vector<Mat> hsv_planes;
	split(img, hsv_planes);*/


	/// Establish the number of bins
	//int histSize = 4;

	int h_bins = 4;
	int s_bins = 3;
	int v_bins = 3;

	int histSize[] = { h_bins, s_bins, v_bins };

	/// Set the ranges ( for H,S,V) )
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 180 };
	float v_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges, v_ranges };


	bool uniform = true; bool accumulate = false;

	Mat h_hist, s_hist, v_hist;
	MatND histhsv;

	/// Compute the histograms:
	//calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, histSize, ranges, uniform, accumulate);
	//calcHist(&hsv_planes[1], 1, 0, Mat(), S_hist, 1, histSize, ranges, uniform, accumulate);
	//calcHist(&hsv_planes[2], 1, 0, Mat(), v_hist, 1, histSize, ranges, uniform, accumulate);

	int channels[] = { 0, 1, 2 };

	calcHist(&img, 1, channels, Mat(), histhsv, 2, histSize, ranges,true,false);

	double maxVal = 0;
	minMaxLoc(histhsv, 0, &maxVal, 0, 0);

	int scale = 10;
	Mat histImg = Mat::zeros(s_bins*scale, h_bins * 10, CV_8UC3);

	for (int h = 0; h < h_bins; h++)
		for (int s = 0; s < s_bins; s++)
		{
			float binVal = histhsv.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(histImg, Point(h*scale, s*scale),Point((h + 1)*scale - 1, (s + 1)*scale - 1),	Scalar::all(intensity),	CV_FILLED);
		}


	// Draw the histograms for B, G and R	
	/*int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}*/

	/// Display
	namedWindow("original", CV_WINDOW_AUTOSIZE);
	imshow("original", src);

	namedWindow("original1", CV_WINDOW_AUTOSIZE);
	imshow("original1", img);

	namedWindow("original5", CV_WINDOW_AUTOSIZE);
	imshow("original5", histImg);

	//namedWindow("original2", CV_WINDOW_AUTOSIZE);
	//imshow("original2", hsv_planes[0]);


	//namedWindow("original3", CV_WINDOW_AUTOSIZE);
	//imshow("original3", hsv_planes[1]);

	//namedWindow("original4", CV_WINDOW_AUTOSIZE);
	//imshow("original4", hsv_planes[2]);
}

void corners(){

	Mat image = imread("img.jpg"), img, imageg;
	int maxc = 20, i = 0;

	vector <Point2d>  pontos;

	cvtColor(image, imageg, CV_RGB2GRAY);

	goodFeaturesToTrack(image, pontos, maxc, 0.01, 10);

	img = image.clone();

	for (i = 0; i < pontos.size(); i++){
		circle(img, pontos[i], 4, CV_RGB(255, 0, 0), -1);
	}

	namedWindow("image original");
	namedWindow("image cantos");

	imshow("image original", image);
	imshow("image cantos", img);

}

void CannyADTVThresh(){
	IplImage *image = cvLoadImage("imagem.jpg", CV_LOAD_IMAGE_GRAYSCALE), *img1 = 0, *img2 = 0;

	img1 = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
	img2 = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1);

	//canny
	cvCanny(image, img1, 10, 10, 3);

	//threshold
	cvAdaptiveThreshold(image, img2, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3, 5.0);

	cvNamedWindow("canny", 1);
	cvNamedWindow("adptative Threshold", 1);

	cvShowImage("canny", img1);
	cvShowImage("adptative Threshold", img2);

}

void BlurTrack(){
	namedWindow("original");
	namedWindow("blurred img");

	imshow("original", image);
	sigma = 0.3*((slider - 1)*0.5 - 1) + 0.8;
	GaussianBlur(image, img, Size(slider, slider), sigma);
	imshow("blurred img", img);
	createTrackbar("Kernel size", "blurred img", &slider, 21, on_trackbar);

	/*"kernel size" - nome do trackbar
	"blurred img" - nome da janela que estara localizado trackbar
	&slider - valor a ser alterado pelo trackbar
	on_trackbar - fc que altera novamente o valor, atualiza-o de acordo com o valor do trackbar*/

}

void ErodeDilate(){
	
	Mat image = imread("imagem.jpg"), st_elem = getStructuringElement(MORPH_RECT, Size(5, 5)), img, img2;

	erode(image, img, st_elem);

	dilate(image, img2, st_elem);

	namedWindow("original");
	namedWindow("erode");

	imshow("original", image);
	imshow("erode", img);

	namedWindow("dilate");
	imshow("dilate", img2);

}

void Blurred(){
	
	Mat image = imread("imagem.jpg"), image_blurred;
	int slider = 5;

	float sigma = 4.1;

	GaussianBlur(image, image_blurred, Size(3,3), 4.1);

	namedWindow("Original");
	namedWindow("blur");

	imshow("Original", image);
	imshow("blur", image_blurred);

}

void FilterKernelMatriz(){
	Mat img1 = imread("imagem.jpg", CV_LOAD_IMAGE_GRAYSCALE), img_filter, img_filter2;


	//matriz para vertical
	//	0   0  0  0  0
	//	0   0  0  0  0
	//	-1 -2  6 -2 -1
	//	0   0  0  0  0
	//	0   0  0  0  0

	float vertical_fk[5][5] = { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { -1, -2, 6, -2, -1 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } };

	float horizontal_fk[5][5] = { { 0, 0, -1, 0, 0 }, { 0, 0, -2, 0, 0 }, { 0, 0, 6, 0, 0 }, { 0, 0, -2, 0, 0 }, { 0, 0, -1, 0, 0 } };


	Mat v_kernel = Mat(5, 5, CV_32FC1, vertical_fk);
	Mat h_kernel = Mat(5, 5, CV_32FC1, horizontal_fk);

	filter2D(img1, img_filter, -1, h_kernel);
	filter2D(img1, img_filter2, -1, v_kernel);

	namedWindow("original");
	namedWindow("filter horizontal");
	namedWindow("filter vertical");
	imshow("original", img1);
	imshow("filter horizontal", img_filter);
	imshow("filter vertical", img_filter2);

}

void InverteImage(){
	cvNamedWindow("picture", 1);

	//IplImage *img1 = cvLoadImage("imagem.jpg"), *img2=cvCreateImage(cvSize(256,256),8,3);

	//int L = img1->width;
	//int A = img1->height;

	int x, y;

	IplImage *img1 = cvLoadImage("imagem.jpg", 1);

	int L = img1->width;
	int A = img1->height;


	IplImage *img2 = cvCreateImage(cvSize(L, A), 8, 3);


	for (y = 0; y < A; y++)
	{
		for (x = 0; x < L; x++)
		{

			int  B = ((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 0];
			int  G = ((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 1];
			int  R = ((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 2];

			((uchar*)(img2->imageData + img2->widthStep*(A - y)))[3 * (L - x) + 0] = B;
			((uchar*)(img2->imageData + img2->widthStep*(A - y)))[3 * (L - x) + 1] = G;
			((uchar*)(img2->imageData + img2->widthStep*(A - y)))[3 * (L - x) + 2] = R;

		}
	}


	cvShowImage("picture", img2);
}

void CorRGB(){
	cvNamedWindow("picture", 1);

	//int L = img1->width;
	//int A = img1->height;

	int x, y;


	IplImage *img1 = cvLoadImage("imagem.jpg", 1);

	int L = img1->width;
	int A = img1->height;


	/*IplImage *img2 = cvCloneImage(img1);*/


	for (y = 0; y < A; y++)
	{
		for (x = 0; x < L; x++)
		{

			int  B = ((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 0];
			int  G = ((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 1];
			int  R = ((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 2];

			int gray = (int)(B + G + R) / 3;
			if (gray < 127){

				((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 0] = 100;
				((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 1] = 100;
				((uchar*)(img1->imageData + img1->widthStep*y))[3 * x + 2] = 100;

			}

		}
	}


	cvShowImage("picture", img1);

}

void RectRoi(){
	
	IplImage *img2 = cvLoadImage("imagem.jpg", 0), *img1 = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);

	int x = img2->width, y = img2->height;

	CvRect ret;
	ret.x = 0;
	ret.y = 0;
	ret.width = 120;
	ret.height = 120;
	img1 = cvCloneImage(img2);

	cvSetImageROI(img1, ret);

	cvNamedWindow("channels img", CV_WINDOW_AUTOSIZE);
	cvShowImage("channels img", img1);

	cvNamedWindow("channels img2", CV_WINDOW_AUTOSIZE);
	cvShowImage("channels img2", img2);


}

void GetSetPixel(){

	int i = 10, j = 10, t = 0;
	IplImage *img = cvLoadImage("imagem.jpg", 3);

	CvScalar s;

	s = cvGet2D(img, i, j);
	// get the (i,j) pixel value

	printf("intensity=%f\n", s.val[0]);

	s.val[0] = 0;
	s.val[1] = 100;
	s.val[2] = 255;
	cvSet2D(img, i, j, s);

	cvSet2D(img, i, j, s);


	cvNamedWindow("channels img1", CV_WINDOW_AUTOSIZE);
	cvShowImage("channels img1", img);
}

void Drawing(){

	IplImage *img = cvLoadImage("imagem.jpg", 1);

	//draw a box
	cvRectangle(img, cvPoint(100, 100), cvPoint(200, 200), cvScalar(0, 255, 0), 1);

	//draw a circle
	cvCircle(img, cvPoint(150, 150), 45, cvScalar(255, 0, 0), 2);

	//draw a line
	cvLine(img, cvPoint(100, 100), cvPoint(200, 200), cvScalar(100, 30, 200), 3);
	cvLine(img, cvPoint(100, 200), cvPoint(200, 100), cvScalar(0, 0, 200), 3);

	//polyline
	CvPoint  curve1[] = { 10, 10, 10, 100, 100, 100, 100, 10 };
	CvPoint  curve2[] = { 30, 30, 30, 130, 130, 130, 130, 30, 150, 10 };
	CvPoint * curvearr[2] = { curve1, curve2 };
	int      ncurvepts[2] = { 4, 5 };
	int      ncurves = 2;
	int      iscurveclosed = 1;
	int      linewidth = 8;

	cvPolyLine(img, curvearr, ncurvepts, ncurves, iscurveclosed, cvScalar(0, 255, 255), linewidth);

	//cvfont font;
	//
	//cvinitfont(&font, cv_font_hershey_simplex | cv_font_italic, 1.0, 1.0, 0, 2);

	//cvputtext(img, "teste", cvpoint(80, 70), &font, cvscalar(255,0, 0));

	//fonts 
	/*cv_font_hershey_simplex, cv_font_hershey_plain,
	cv_font_hershey_duplex, cv_font_hershey_complex,
	cv_font_hershey_triplex, cv_font_hershey_complex_small,
	cv_font_hershey_script_simplex, cv_font_hershey_script_complex,*/


	cvNamedWindow("drawing", CV_WINDOW_AUTOSIZE);
	cvShowImage("drawing", img);

}

void mask(){
	Mat image = imread("mhorizonte.png"), mask = imread("mask.png"), imagefinal= Mat::zeros(3,3,CV_8UC1);

	image.copyTo(imagefinal, mask);

	imshow("final", imagefinal);
	imshow("imagem", image);
	imshow("mask", mask);

//	HistogramaMask(imagefinal);


}

void HistogramaMask_(Mat image, int h, int s, int v){

	FILE* arq = fopen("maskhistograma.txt", "w");

	vector<int> hist;

	Mat img_hsv, HSV;

	float ph = 0, ps = 0, pv = 0;

	//scanf("%d %d %d", &h, &s, &v);

	ph = 360.0 / (h - 1);
	ps = 255.0 / (s - 1);
	pv = 255.0 / (v - 1);


	hist.resize((h*s*v), 0);

	cvtColor(image, img_hsv, CV_BGR2HSV_FULL);
	//Convert RGB to HSV (FULL)->(0 - 360, 0-255, 0-255)

	//namedWindow("original");
	//namedWindow("hsv");

	//imshow("original", image);
	//imshow("hsv", img_hsv);

	//cvWaitKey(0);

	int k = 0, t = 0, h1 = 0, s1 = 0, v1 = 0, teste = 0;

	for (int x = 0; x < img_hsv.rows; x++){

		for (int y = 0; y < img_hsv.cols; y++){

			Vec3b p = img_hsv.at<Vec3b>(x, y);

			h1 = p.val[0];
			s1 = p.val[1];
			v1 = p.val[2];

			//printf("H=%d, S=%d, V=%d\n", h1, s1, v1);
			fprintf(arq, "H=%d, S=%d, V=%d\n", h1, s1, v1);

			if (h1 != 0 && s1 != 0 && v1 != 0){

				//printf("H(entravetor)=%d, S(entravetor)=%d, V(entravetor)=%d\n", h1, s1, v1);
				fprintf(arq, "H(entravetor)=%d, S(entravetor)=%d, V(entravetor)=%d\n", h1, s1, v1);

				particiona(&h1, ph);
				particiona(&s1, ps);
				particiona(&v1, pv);

				k = ((s*v)*h1) + (v*s1) + v1;
				hist[k] += 1;

				k = 0;
				t++;
			}

		}
	}

	//printf("%d\n", t);
	fprintf(arq, "quantidade pixel que entrou no vetor:%d\n", t);

	//printf("vetor:\n");
	fprintf(arq, "Vetor: \n");

	for (int i = 0; i < (h*s*v); i++){
		//	printf("%d ", hist[i]);
		fprintf(arq, "%d ", hist[i]);
	}

	fclose(arq);
}

vector<float> normalizavetor(vector<float>hist, int t, int size){
	
	t = (float)t;
	int i = 0;
	for (i = 0; i < size; i++){
		//printf("antes: %f", hist[i]);

		hist[i] = hist[i]/t;

		//printf("depois: %f", hist[i]);
	}

	return hist;
}

void HistogramaMask(char *nomearquivo, Mat image, Mat mask, int h, int s, int v){

	FILE* arq = fopen(nomearquivo, "w");

	vector<float> hist;

	Mat img_hsv, HSV;

	float ph = 0, ps = 0, pv = 0;

	//scanf("%d %d %d", &h, &s, &v);

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
		
			/*uchar m = mask.at<uchar>(x, y);
			fprintf(arq, "\n\nimage mascara = %d\n",m);*/

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
	
	//ContornoHullMAT();
	
	char *nomeimage = argv[1], *nomemask = argv[2],*nomearquivo = argv[3];
	int arg_h = atoi(argv[4]), arg_s = atoi(argv[5]), arg_v = atoi(argv[6]);
	

	//printf("%d %d %d", arg_h, arg_s, arg_v);

	Mat image = imread(nomeimage), mask = imread(nomemask), final;

	//final = ApplymaskHist(image, mask);
	
	HistogramaMask(nomearquivo, image, mask, arg_h, arg_s, arg_v);

	//system("pause");
	cvWaitKey(0);
	cvDestroyAllWindows();

	return 0;
}

