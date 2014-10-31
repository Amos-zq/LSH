/*
Visual Tracking via Locality Sensitive Histograms
Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, Ming-Hsuan Yang
Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013
*/
#ifndef HE_LSH_H
#define HE_LSH_H
#include "He_basic.h"

using namespace cv;
#define feature_bin_size 64
#define hist_bin_size 32
const int feature_QTZ = 256/feature_bin_size;
const int QTZ = 256/hist_bin_size;
#define down_scale 1

void LSH_recursive(double**out,Mat in,double**temp,double**temp_2w, double alphax,double alphay, int h,int w, int k
								  , Point hist_lt, Point hist_rb)//temp_2w [2,w], alpha 0.6-0.95
{
	/*horizontal*/
	double**in_;
	double*temp_x=temp_2w[0];
	double *optr=out[k];
	for(int y=hist_lt.y;y<hist_rb.y;y++)
	{
		uchar* in_ptr = in.ptr<uchar>(y);
		double*tptr = temp[y];
		tptr[0]=((in_ptr[0])==k);
		double yc,yp=((in_ptr[0])==k);
		for(int x=hist_lt.x;x<hist_rb.x;x++)
		{
			tptr[x]=yc=(1-alphax)*((in_ptr[x])==k)+alphax*yp;
			yp=yc;
		}
		int w1=hist_rb.x-1;
		yp=((in_ptr[w1])==k);
		tptr[w1]=0.5*(tptr[w1]+yp);
		for(int x=hist_rb.x-2;x>=hist_lt.x;x--)
		{
			yc=(1-alphax)*((in_ptr[x])==k)+alphax*yp;
			tptr[x]=0.5*(tptr[x]+yc);
			yp=yc;
		}
	}

	/*vertical*/
	in_=temp;
	double*ycy,*ypy,*xcy;
	memcpy(optr,temp[0],sizeof(double)*w);
	for(int y=hist_lt.y+1;y<hist_rb.y;y++)
	{
		xcy=in_[y];
		ypy=optr + (y-1)*w;
		ycy=optr + y*w;
		for(int x=hist_lt.x;x<hist_rb.x;x++)
		{
			ycy[x]=(1-alphay)*xcy[x]+alphay*ypy[x];
		}
	}
	int h1=h-1;
	ycy=temp_2w[0];
	ypy=temp_2w[1];
	memcpy(ypy,in_[h1],sizeof(double)*w);
	for(int x=hist_lt.x;x<hist_rb.x;x++) optr[x+h1*w]=0.5*(optr[x+h1*w]+ypy[x]);
	for(int y=hist_rb.y-2;y>=hist_lt.y;y--)
	{
		xcy=in_[y];//optr=out[y];
		for(int x=hist_lt.x;x<hist_rb.x;x++)
		{
			ycy[x]=(1-alphay)*xcy[x]+alphay*ypy[x];
			optr[x + y*w]=0.5*(optr[x+y*w]+ycy[x]);
		}
		memcpy(ypy,ycy,sizeof(double)*w);
	}
}

void IIF_LSH_recursive(double**out,Mat in,double**temp,double**temp_2w, double alphax,double alphay, int h,int w, int k
				   , Point hist_lt, Point hist_rb)//temp_2w [2,w], alpha 0.6-0.95
{
	/*horizontal*/
	double**in_;
	double*temp_x=temp_2w[0];
	double *optr=out[k];
	for(int y=hist_lt.y;y<hist_rb.y;y++)
	{
		uchar* in_ptr = in.ptr<uchar>(y);
		double*tptr = temp[y];
		tptr[0]=((in_ptr[0])==k);
		double yc,yp=((in_ptr[0])==k);
		for(int x=hist_lt.x;x<hist_rb.x;x++)
		{
			tptr[x]=yc=(1-alphax)*((in_ptr[x])==k)+alphax*yp;
			yp=yc;
		}
		int w1=hist_rb.x-1;
		yp=((in_ptr[w1])==k);
		tptr[w1]=0.5*(tptr[w1]+yp);
		for(int x=hist_rb.x-2;x>=hist_lt.x;x--)
		{
			yc=(1-alphax)*((in_ptr[x])==k)+alphax*yp;
			tptr[x]=0.5*(tptr[x]+yc);
			yp=yc;
		}
	}

	/*vertical*/
	in_=temp;
	double*ycy,*ypy,*xcy;
	memcpy(optr,temp[0],sizeof(double)*w);
	for(int y=hist_lt.y+1;y<hist_rb.y;y++)
	{
		xcy=in_[y];
		ypy=optr + (y-1)*w;
		ycy=optr + y*w;
		for(int x=hist_lt.x;x<hist_rb.x;x++)
		{
			ycy[x]=(1-alphay)*xcy[x]+alphay*ypy[x];
		}
	}
	int h1=h-1;
	ycy=temp_2w[0];
	ypy=temp_2w[1];
	memcpy(ypy,in_[h1],sizeof(double)*w);
	for(int x=hist_lt.x;x<hist_rb.x;x++) optr[x+h1*w]=0.5*(optr[x+h1*w]+ypy[x]);
	for(int y=hist_rb.y-2;y>=hist_lt.y;y--)
	{
		xcy=in_[y];//optr=out[y];
		for(int x=hist_lt.x;x<hist_rb.x;x++)
		{
			ycy[x]=(1-alphay)*xcy[x]+alphay*ypy[x];
			optr[x + y*w]=0.5*(optr[x+y*w]+ycy[x]);
		}
		memcpy(ypy,ycy,sizeof(double)*w);
	}
}


inline void Compute_LSH(double**histogram, Mat in, double**histogram_temp,double**temp3w, double sigma,int h,int w, Point hist_lt, Point hist_rb)
{
	Mat idx_img = in/QTZ;
	double t;
	{
		double alphay=exp(-sqrt(2.0)/(sigma*h));
		double alphax=exp(-sqrt(2.0)/(sigma*w));

		t = (double)cvGetTickCount();	
		for(int k = 0; k< hist_bin_size; k++)
		{
			LSH_recursive(histogram, idx_img, histogram_temp, temp3w, alphax,alphay, h, w, k, hist_lt, hist_rb);			
		}

		t = (double)cvGetTickCount() - t;
	}
}

inline void Compute_IIF_LSH(double**histogram, Mat in, double**histogram_temp,double**temp3w, double sigma,int h,int w, Point hist_lt, Point hist_rb)
{
	Mat idx_img = in/feature_QTZ;
	double t;
	{
		double alphay=exp(-sqrt(2.0)/(sigma*h));
		double alphax=exp(-sqrt(2.0)/(sigma*w));

		t = (double)cvGetTickCount();	
		for(int k = 0; k< feature_bin_size; k++)
		{
			IIF_LSH_recursive(histogram, idx_img, histogram_temp, temp3w, alphax,alphay, h, w, k, hist_lt, hist_rb);			
		}

		t = (double)cvGetTickCount() - t;
	}
}
#endif