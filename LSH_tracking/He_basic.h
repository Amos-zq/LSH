#ifndef HE_BASIC_H
#define HE_BASIC_H
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cv.h"
#include "cxcore.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <numeric>
#include <vector>
#include <process.h>
#include <direct.h>
#include <io.h>
#include <time.h>
#include <string>
#include <memory.h>
#include <algorithm>
#include <functional>      // For greater<int>()
#include <iostream>
//#include <imdebug.h>
using namespace std;
#define HE_DEF_THRESHOLD_ZERO			1e-6
#define HE_DEF_PADDING					0
class   he_timer		{public: void start();	float stop(); void time_display(char *disp=""); void fps_display(char *disp="",int nr_frame=1); private: clock_t m_begin; clock_t m_end;};
/*Box filter*/
//void boxcar_sliding_window(double **out,double **in,double **temp,int h,int w,int radius);
//void boxcar_sliding_window_x(double *out,double *in,int h,int w,int radius);
//void boxcar_sliding_window_y(double *out,double *in,int h,int w,int radius);
/*Gaussian filter*/
//int gaussian_recursive(double **image,double **temp,double sigma,int order,int h,int w);
//void gaussian_recursive_x(double **od,double **id,int w,int h,double a0,double a1,double a2,double a3,double b1,double b2,double coefp,double coefn);
//void gaussian_recursive_y(double **od,double **id,int w,int h,double a0,double a1,double a2,double a3,double b1,double b2,double coefp,double coefn);
/*basic functions*/
inline double *get_color_weighted_table(double sigma_range,int len)
{
	double *table_color,*color_table_x; int y;
	table_color=new double [len];
	color_table_x=&table_color[0];
	for(y=0;y<len;y++) (*color_table_x++)=exp(-double(y*y)/(2*sigma_range*sigma_range));
	return(table_color);
}
/*memory*/
inline double *** he_allocd_3(int n,int r,int c,int padding=HE_DEF_PADDING)
{
	double *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(double*) malloc(sizeof(double)*(n*rc+padding));
	if(a==NULL) {printf("he_allocd_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(double**) malloc(sizeof(double*)*n*r);
    pp=(double***) malloc(sizeof(double**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}
inline void he_freed_3(double ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline unsigned char** he_allocu(int r,int c,int padding=HE_DEF_PADDING)
{
	unsigned char *a,**p;
	a=(unsigned char*) malloc(sizeof(unsigned char)*(r*c+padding));
	if(a==NULL) {printf("he_allocu() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(unsigned char**) malloc(sizeof(unsigned char*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void he_freeu(unsigned char **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline unsigned char *** he_allocu_3(int n,int r,int c,int padding=HE_DEF_PADDING)
{
	unsigned char *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(unsigned char*) malloc(sizeof(unsigned char )*(n*rc+padding));
	if(a==NULL) {printf("he_allocu_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(unsigned char**) malloc(sizeof(unsigned char*)*n*r);
    pp=(unsigned char***) malloc(sizeof(unsigned char**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}
inline void he_freeu_3(unsigned char ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline float** he_allocf(int r,int c,int padding=HE_DEF_PADDING)
{
	float *a,**p;
	a=(float*) malloc(sizeof(float)*(r*c+padding));
	if(a==NULL) {printf("he_allocf() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(float**) malloc(sizeof(float*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void he_freef(float **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline float *** he_allocf_3(int n,int r,int c,int padding=HE_DEF_PADDING)
{
	float *a,**p,***pp;
    int rc=r*c;
    int i,j;
	a=(float*) malloc(sizeof(float)*(n*rc+padding));
	if(a==NULL) {printf("he_allocf_3() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
    p=(float**) malloc(sizeof(float*)*n*r);
    pp=(float***) malloc(sizeof(float**)*n);
    for(i=0;i<n;i++) 
        for(j=0;j<r;j++) 
            p[i*r+j]=&a[i*rc+j*c];
    for(i=0;i<n;i++) 
        pp[i]=&p[i*r];
    return(pp);
}
inline void he_freef_3(float ***p)
{
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline int** he_alloci(int r,int c,int padding=HE_DEF_PADDING)
{
	int *a,**p;
	a=(int*) malloc(sizeof(int)*(r*c+padding));
	if(a==NULL) {printf("he_alloci() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(int**) malloc(sizeof(int*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void he_freei(int **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}
inline double** he_allocd(int r,int c,int padding=HE_DEF_PADDING)
{
	double *a,**p;
	a=(double*) malloc(sizeof(double)*(r*c+padding));
	if(a==NULL) {printf("he_allocd() fail, Memory is too huge, fail.\n"); getchar(); exit(0); }
	p=(double**) malloc(sizeof(double*)*r);
	for(int i=0;i<r;i++) p[i]= &a[i*c];
	return(p);
}
inline void he_freed(double **p)
{
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}

inline double he_linear_interpolate_xy(double **image,double x,double y,int h,int w)
{
	int x0,xt,y0,yt; double dx,dy,dx1,dy1,d00,d0t,dt0,dtt;
	x0=int(x); xt=min(x0+1,w-1); y0=int(y); yt=min(y0+1,h-1);
	dx=x-x0; dy=y-y0; dx1=1-dx; dy1=1-dy; d00=dx1*dy1; d0t=dx*dy1; dt0=dx1*dy; dtt=dx*dy;
	return(d00*image[y0][x0]+d0t*image[y0][xt]+dt0*image[yt][x0]+dtt*image[yt][xt]);
}

// inline void image_display(int *in,int h,int w)
// {
// 	int i,len=h*w; float *disp;
// 	disp=new float [len];
// 	for(i=0;i<len;i++) disp[i]=(float)in[i];
// 	imdebug("lum *auto b=32f w=%d h=%d %p",w,h,disp);
// 	delete [] disp; disp=NULL;
// }
// inline void image_display(unsigned char **in,int h,int w)
// {
// 	imdebug("lum *auto w=%d h=%d %p",w,h,&in[0][0]);
// }
// inline void image_display(float **in,int h,int w)
// {
// 	imdebug("lum *auto b=32f w=%d h=%d %p",w,h,&in[0][0]);
// }
// inline void image_display(float **in,int h,int w,float threshold)
// {
// 	float *disp,*disp_in; int len,i;
// 	len=h*w;
// 	disp=new float [len];
// 	disp_in=in[0];
// 	for(i=0;i<len;i++) if(disp_in[i]>=threshold) disp[i]=1; else disp[i]=0;
// 	imdebug("lum *auto b=32f w=%d h=%d %p",w,h,disp);
// 	delete [] disp; disp=NULL;
// }
// inline void image_display(unsigned short **in,int h,int w)
// {
// 	float **disp=he_allocf(h,w);
// 	for(int y=0;y<h;y++) for(int x=0;x<w;x++) disp[y][x]=(float)in[y][x];
// 	imdebug("lum *auto b=32f w=%d h=%d %p",w,h,&disp[0][0]);
// 	he_freef(disp); disp=NULL;
// }
// inline void image_display(int **in,int h,int w)
// {
// 	float **disp=he_allocf(h,w);
// 	for(int y=0;y<h;y++) for(int x=0;x<w;x++) disp[y][x]=(float)in[y][x];
// 	imdebug("lum *auto b=32f w=%d h=%d %p",w,h,&disp[0][0]);
// 	he_freef(disp); disp=NULL;
// }
// inline void image_display(long int **in,int h,int w)
// {
// 	float **disp=he_allocf(h,w);
// 	for(int y=0;y<h;y++) for(int x=0;x<w;x++) disp[y][x]=(float)in[y][x];
// 	imdebug("lum *auto b=32f w=%d h=%d %p",w,h,&disp[0][0]);
// 	he_freef(disp); disp=NULL;
// }
// inline void image_display(unsigned char ***in,int h,int w)
// {
// 	imdebug("rgb *auto w=%d h=%d %p",w,h,&in[0][0][0]);
// }
// inline void image_display(float ***in,int h,int w)
// {
// 	imdebug("rgb *auto b=32f w=%d h=%d %p",w,h,&in[0][0][0]);
// }
// inline void image_display(double ***in,int h,int w)
// {
// 	float ***out=he_allocf_3(h,w,3);
// 	for(int y=0;y<h;y++)
// 	{
// 		for(int x=0;x<w;x++) 
// 		{
// 			out[y][x][0]=(float)in[y][x][0];
// 			out[y][x][1]=(float)in[y][x][1];
// 			out[y][x][2]=(float)in[y][x][2];
// 		}
// 	}
// 	imdebug("rgb *auto b=32f w=%d h=%d %p",w,h,&out[0][0][0]);
// 	he_freef_3(out); out=NULL;
// }
// inline void image_display_2(float ***in,int h,int w)
// {
// 	float ***out=he_allocf_3(h,w,3);
// 	for(int y=0;y<h;y++)
// 	{
// 		for(int x=0;x<w;x++) 
// 		{
// 			out[y][x][0]=in[y][x][0];
// 			out[y][x][1]=in[y][x][1];
// 			out[y][x][2]=0;
// 		}
// 	}
// 	imdebug("rgb *auto b=32f w=%d h=%d %p",w,h,&out[0][0][0]);
// 	he_freef_3(out); out=NULL;
// }
// inline void image_display_2(int ***in,int h,int w)
// {
// 	float ***out=he_allocf_3(h,w,3);
// 	for(int y=0;y<h;y++)
// 	{
// 		for(int x=0;x<w;x++) 
// 		{
// 			out[y][x][0]=(float)in[y][x][0];
// 			out[y][x][1]=(float)in[y][x][1];
// 			out[y][x][2]=0;
// 		}
// 	}
// 	imdebug("rgb *auto b=32f w=%d h=%d %p",w,h,&out[0][0][0]);
// 	he_freef_3(out); out=NULL;
// }
// inline void image_display_4(unsigned char ***in,int h,int w){imdebug("rgba *auto w=%d h=%d %p",w,h,&in[0][0][0]);}
// inline void image_display_4(unsigned char *in,int h,int w){imdebug("rgba *auto w=%d h=%d %p",w,h,in);}
// inline void image_display_4(float *in,int h,int w){imdebug("rgba *auto b=32f w=%d h=%d %p",w,h,in);}
// inline void image_display_4(short*in,int h,int w){imdebug("rgba *auto b=16 w=%d h=%d %p",w,h,in);}
// inline void image_display_3(unsigned char *in,int h,int w){imdebug("rgb *auto w=%d h=%d %p",w,h,in);}
// inline void image_display_3(float *in,int h,int w){imdebug("rgb *auto b=32f w=%d h=%d %p",w,h,in);}
// inline void image_display(unsigned char *in,int h,int w){imdebug("lum *auto w=%d h=%d %p",w,h,in);}
// inline void image_display(float *in,int h,int w){imdebug("lum *auto b=32f w=%d h=%d %p",w,h,in);}
// inline void image_display(short*in,int h,int w){imdebug("lum *auto b=16 w=%d h=%d %p",w,h,in);}
// inline void image_display(short**in,int h,int w){imdebug("lum *auto b=16 w=%d h=%d %p",w,h,in[0]);}
// inline void image_display_rgba(float ***in,int h,int w)
// {
// 	imdebug("rgba *auto b=32f w=%d h=%d %p",w,h,&in[0][0][0]);
// }
// inline void image_display(double **in,int h,int w)
// {
// 	float **disp=he_allocf(h,w);
// 	for(int y=0;y<h;y++) for(int x=0;x<w;x++) disp[y][x]=(float)in[y][x];
// 	imdebug("lum *auto b=32f w=%d h=%d %p",w,h,&disp[0][0]);
// 	he_freef(disp); disp=NULL;
// }
inline unsigned char dist_rgb(unsigned char *a,unsigned char *b)
{
	float out; 
	out=float(abs(a[0]-b[0])+abs(a[1]-b[1])+abs(a[2]-b[2])); 
	return(unsigned char(out/3+0.5f));
}
void down_sample_1(unsigned char **out,unsigned char **in,int h,int w,int scale_exp);
void rgb_2_gray(unsigned char**out,unsigned char***in,int h,int w);
inline void color_weighted_table_update(double *table_color,double dist_color,int len)
{
	double *color_table_x; int y;
	color_table_x=&table_color[0];
	for(y=0;y<len;y++) (*color_table_x++)=exp(-double(y*y)/(2*dist_color*dist_color));
}
inline void vec_max_val(unsigned char &max_val,unsigned char *in,int len)
{
	max_val=in[0];
	for(int i=1;i<len;i++) if(in[i]>max_val) max_val=in[i];	
}
inline void vec_min_val(unsigned char &min_val,unsigned char *in,int len)
{
	min_val=in[0];
	for(int i=1;i<len;i++) if(in[i]<min_val) min_val=in[i];	
}

#endif