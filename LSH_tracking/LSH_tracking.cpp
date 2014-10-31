/*
Visual Tracking via Locality Sensitive Histograms
Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, Ming-Hsuan Yang
Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013
*/

#include "He_basic.h"
#include "LSH.h"
#include<fstream>
#include <iostream>

using namespace std;
using namespace cv;

Mat frame;Mat temp_image;Mat img;
Mat out; Mat grayimg; Mat grayimg_scale;Mat target_img; Mat target_gray; Mat target_gray_scale;
int target_height, target_width, imgheight, imgwidth;
int region_num_x, region_num_y;
Point lt,rb,clt, current_center, new_center, new_lt;
int track_object=0;
Point origin;
Rect selection, new_rect;

//parameters, note that these parameters will be re-defined by the initial text file, e.g. david_indoor.txt
float LSH_Spatial_sigma = 0.015;	//Histogram coefficient
float IIF_Spatial_sigma = 0.023;	//feature coefficient (is not the same histograms with above)

int search_radius = 8;				//tracking search region
float region_space = 2.0;			//the space between each region
int detect_per = 4;					//the percentage of score (e.g. 4 means take 25% of the total score as the final score)
int Fast_IIF_radius = 1;				//feature range
bool feature_mode = 1;					//feature we used, 0=intensity,1=ours
float ff = 0.04;					//forgetting factor
int ex_region = 15;					//LSH computing radius
bool visualize = 1;
//parameters, note that these parameters will be re-defined by the initial text file, e.g. david_indoor.txt
int region_num;

#define  gau_radius 512

float *gaussian = new float[gau_radius*2];
float iif_table[256][256];

void updateGaussiantable(float delta, int radius)
{
	for(int i = 0; i < 2 * radius; i++)
	{
		int x = i - radius;
		gaussian[i] = -(-2*x/(2 * delta*delta ))*exp(-(x * x) / (2 * delta*delta ));
	}
}

void updateiif_table(float delta)
{
	double r,kappa,rr;
	for(int i = 0; i < 256; i++){
		kappa = (delta);
		r = kappa*max(double(i),1.0);
		rr = max(kappa,r);		
		for(int j =0;j<256;j++)
		{
			iif_table[i][j] = exp(-(j)*(j)/(2*rr*rr));
		}
	}
}

void Feature_extraction(Mat id, Mat od, double**current_hist, int w, int h)
{
	double **hsum = he_allocd(h, w);
	printf( "Feature_Extraction...");

	for(int k = 0;k<feature_bin_size;k++){
		for(int i=0;i<h;i++){
			uchar* id_ptr = id.ptr<uchar>(i);
			for(int j=0;j<w;j++)
			{
				if(k==0)hsum[i][j]=0;
				double weight;
				int cint = id_ptr[j];

				int downvalue = abs(int(cint));

				double i_tmp = k-(double)cint/feature_QTZ;

				weight = iif_table[downvalue][int(abs(i_tmp))];
				hsum[i][j] += current_hist[k][j/down_scale+i/down_scale*w/down_scale]*weight;
			}
		}
	}
	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){	od.ptr<uchar>(i)[j] = max(min(hsum[i][j],1.0),0.0)*255;}}
	he_freed(hsum);
}

int patition(double *a,int p,int r)  
{  
	double x=a[r];  
	int i=p-1;  
	int j;  
	for(j=p;j<=r-1;j++)  
	{  
		if(a[j]<=x)  
		{  
			i++;  
			double temp=a[j];  
			a[j]=a[i];  
			a[i]=temp;  
		}  
	}  
	double tem;  
	tem=a[i+1];  
	a[i+1]=a[r];  
	a[r]=tem;  
	return i+1;  
}  
double randomselect(double *a,int p,int r,int i)  
{  
	if(p==r)  
		return a[p];  
	int q=patition(a,p,r);  
	int k=q-p+1;  
	if(i==k)  
		return a[q];  
	else if(i<k)  
		return randomselect(a,p,q-1,i);
	else   
		return randomselect(a,q+1,r,i-k);  
}  

double Histogram_distance_EMD(double*current, double*selet)
{
	double d_result = 0;
	double dif;
	for(int k=0;k<hist_bin_size;k++)
	{
		dif = abs(current[k]-selet[k]);
		d_result += (dif);
	}

	return d_result;
}

double Histogram_distance_naive(double**current, double**selet, Point current_pos, Point selet_pos, int Hist_width)
{
	double d_result = 0;
	double dif;
	for(int k=0;k<hist_bin_size;k++)
	{
		dif = abs(current[current_pos.x + current_pos.y*Hist_width][k]-selet[selet_pos.x+selet_pos.y*Hist_width][k]);
		d_result += (dif);
	}

	return d_result;
}

void Hist_2_CumuHist(double**Hist, double*CumuHist, Point Hist_pos, int Hist_width)
{
	for(int k=0;k<hist_bin_size;k++)
	{
		if(k==0)
			CumuHist[k] = Hist[Hist_pos.x+Hist_pos.y*Hist_width][k];
		else
			CumuHist[k] = CumuHist[k-1] + Hist[Hist_pos.x+Hist_pos.y*Hist_width][k];
	}
}

void hist_transpose(double**Hist, double**new_Hist, int w, int h)
{
	for(int i = 0;i<h;i++){
		for (int j = 0;j<w;j++){
			for(int k=0;k<hist_bin_size;k++)
			{
				new_Hist[j+i*w][k] = Hist[k][j+i*w];
			}
		}
	}
}

int main(int argc, char * argv[])
{
	char *v_name = (argv[1]);
	char *v_path = argv[2];
	int frame_start = atoi(argv[3]);
	int frame_end = atoi(argv[4]);
	int img_str_length = atoi(argv[5]);
	char *ext_name = argv[6];

	std::stringstream temp_name;
	temp_name <<std::setfill('0')<<std::setw(img_str_length)<< frame_start;


	char frame_name[255];
	sprintf_s(frame_name,255,"%s\img%s.%s",v_path,temp_name.str().c_str(),ext_name);

	cout<<endl<<endl;
	puts(frame_name);
	cout<<endl<<endl;

	char output_position[255];

	strcpy(output_position,"");
	sprintf(output_position,"%s.txt",v_name);
	ofstream outfile(output_position);	
	strcpy(output_position,"");
	sprintf(output_position,"%s_FPS.txt",v_name);
	ofstream outFPS(output_position);

	frame=imread(frame_name);
	if(!frame.data)
	{
		printf("Image is unavailable.\n");
		return -1;
	}
	int imgheight = frame.rows;
	int imgwidth = frame.cols;

	float init_para[11];	

	init_para[0] = atoi(argv[7]);
	init_para[1] = atoi(argv[8]);
	init_para[2] = atoi(argv[9]);
	init_para[3] = atoi(argv[10]);

	cout<<"Parameters:"<<endl;
	feature_mode = atoi(argv[11]);
	printf("Feature = %d\n",feature_mode);
	IIF_Spatial_sigma = atof(argv[12]);
	printf("Alpha = %f\n",IIF_Spatial_sigma);
	search_radius = atoi(argv[13]);
	printf("Search region = %d\n",search_radius);
	region_space = 2;
	ff = atof(argv[14]);
	visualize = atof(argv[15]);
	printf("forgetting factors = %f\n",ff);
	cout<<endl<<"Start:"<<endl;

	if(visualize){
		namedWindow( "Feature", CV_WINDOW_AUTOSIZE ); 
		namedWindow( "Tracking", CV_WINDOW_AUTOSIZE ); }

	out.create(imgheight, imgwidth, CV_8U);
	double** current_hist = he_allocd(hist_bin_size, (imgheight/down_scale)*(imgwidth/down_scale));
	double** current_hist_trans = he_allocd((imgheight/down_scale)*(imgwidth/down_scale), hist_bin_size);
	double** iif_current_hist = he_allocd(feature_bin_size, (imgheight/down_scale)*(imgwidth/down_scale));
	double** template_hist = he_allocd(hist_bin_size, (imgheight/down_scale)*(imgwidth/down_scale));
	double** template_hist_trans = he_allocd((imgheight/down_scale)*(imgwidth/down_scale), hist_bin_size);
	double** histogram_temp = he_allocd(imgheight/down_scale, imgwidth/down_scale);
	double** temp3w = he_allocd(3, imgwidth/down_scale);
	double* hist_weight = new double[hist_bin_size];
	double* template_cumu_hist = new double[hist_bin_size];
	double* new_cumu_hist = new double[hist_bin_size];
	double* distance_storage = new double[imgheight/down_scale*imgwidth/down_scale];
	double* best_distance_storage = new double[imgheight/down_scale*imgwidth/down_scale];
	double* temp_distance_storage = new double[imgheight/down_scale*imgwidth/down_scale];
	if(visualize)
		imshow( "Tracking", frame );
	img=frame.clone();
	//updateGaussiantable(weight_sigma, gau_radius);
	updateiif_table(IIF_Spatial_sigma);

	/**********************Compute Template****************************/
	Point hist_lt = Point(0,0), hist_rb = Point(imgwidth/down_scale,imgheight/down_scale);

	selection.x = init_para[0];
	selection.y = init_para[1];
	selection.width = init_para[2];
	selection.height = init_para[3];

	cvtColor( frame, grayimg, CV_BGR2GRAY );
	resize(grayimg, grayimg_scale, Size(), 1/double(down_scale),1/double(down_scale));

	current_center = Point(selection.x+selection.width/2,selection.y+selection.height/2);
	lt = Point(selection.x,selection.y);
	lt.x = max(min(lt.x,imgwidth-1),0);
	lt.y = max(min(lt.y,imgheight-1),0);
	getRectSubPix(frame, Size(selection.width,selection.height), current_center, target_img);
	imwrite("template.jpg", target_img);

// 	/**********************LSH region****************************/
// 	hist_lt = Point(lt.x/down_scale,lt.y/down_scale) - Point(search_radius*2,search_radius*2) - Point(ex_region,ex_region);
// 	hist_lt.x = max(hist_lt.x,0);hist_lt.y = max(hist_lt.y,0);
// 
// 	hist_rb = Point(lt.x/down_scale,lt.y/down_scale) + Point(search_radius*2,search_radius*2) + Point(selection.width/down_scale,selection.height/down_scale) + Point(ex_region,ex_region);
// 	hist_rb.x = min(hist_rb.x,imgwidth/down_scale);hist_rb.y = min(hist_rb.y,imgheight/down_scale);
// 	/**********************LSH region****************************/

	if(feature_mode){
		Compute_IIF_LSH(iif_current_hist, grayimg_scale, histogram_temp, temp3w, IIF_Spatial_sigma, imgheight/down_scale, imgwidth/down_scale, hist_lt,hist_rb);
		Feature_extraction(grayimg, out, iif_current_hist, imgwidth, imgheight);
		resize(out, grayimg_scale, Size(), 1/double(down_scale),1/double(down_scale));
	}

	Compute_LSH(template_hist, grayimg_scale, histogram_temp, temp3w, LSH_Spatial_sigma, imgheight/down_scale, imgwidth/down_scale, hist_lt,hist_rb);
	hist_transpose(template_hist, template_hist_trans, imgwidth/down_scale, imgheight/down_scale);
	outfile<<lt.x<<' '<<lt.y<<' '<<selection.width<<' '<<selection.height<<' '<<endl;
	/**********************Compute Template****************************/

	float avgFPS = 0;int num_f = 0;

	int framenum = frame_start+1;

	/**********************Start tracking from the 2nd frame****************************/
	while(selection.width!=0&&selection.height!=0) 
	{
		cout<<"Tracking..."<<v_name<<endl;
		printf("%dth frame\n",framenum);
		if( !frame.data ||framenum>frame_end) 
		{break;}
		temp_name.str("");
		temp_name <<std::setfill('0')<<std::setw(img_str_length)<< framenum;
		char frame_name[255];
		sprintf_s(frame_name,255,"%s\img%s.%s",v_path,temp_name.str().c_str(),ext_name);
		frame=imread(frame_name);
		framenum++;
		if(visualize)
			imshow( "Tracking", frame );
		img=frame.clone();
				
		cvtColor( frame, grayimg, CV_BGR2GRAY );
		resize(grayimg, grayimg_scale, Size(), 1/double(down_scale),1/double(down_scale));

		/**********************LSH region****************************/
		hist_lt = Point(lt.x/down_scale,lt.y/down_scale) - Point(search_radius*2,search_radius*2) - Point(ex_region,ex_region);
		hist_lt.x = max(hist_lt.x,0);hist_lt.y = max(hist_lt.y,0);

  		hist_rb = Point(lt.x/down_scale,lt.y/down_scale) + Point(search_radius*2,search_radius*2) + Point(selection.width/down_scale,selection.height/down_scale) + Point(ex_region,ex_region);
  		hist_rb.x = min(hist_rb.x,imgwidth/down_scale);hist_rb.y = min(hist_rb.y,imgheight/down_scale);
  		rectangle(img, hist_lt*down_scale, hist_rb*down_scale, CV_RGB(255,128,12),1);
		/**********************LSH region****************************/

		double t = (double)cvGetTickCount();

		/**********************Compute LSH for IIF****************************/
		if(feature_mode){
			Compute_IIF_LSH(iif_current_hist, grayimg_scale, histogram_temp, temp3w, IIF_Spatial_sigma, imgheight/down_scale, imgwidth/down_scale, hist_lt,hist_rb);
			Feature_extraction(grayimg, out, iif_current_hist, imgwidth, imgheight);
			resize(out, grayimg_scale, Size(), 1/double(down_scale),1/double(down_scale));
		}
		/**********************Compute LSH for IIF****************************/

		/**********************Compute LSH for tracking****************************/
		Compute_LSH(current_hist, grayimg_scale, histogram_temp, temp3w, LSH_Spatial_sigma, imgheight/down_scale, imgwidth/down_scale, hist_lt,hist_rb);
		hist_transpose(current_hist, current_hist_trans, imgwidth/down_scale, imgheight/down_scale);
		/**********************Compute LSH for tracking****************************/

		int counterx=0,countery=0;
		double accuvalue = 0;double meanvalue = 0;

		/**********************Tracking****************************/
		double lowest_dist=10,new_dist=0;

		for(int i=-search_radius;i<search_radius;i++){
			for(int j=-search_radius;j<search_radius;j++)
			{
				int k = 0;accuvalue = 0;
				for(float y=region_space/2;y<selection.height/down_scale-region_space/2;y+=region_space)
				{
					for(float x=region_space/2;x<selection.width/down_scale-region_space/2;x+=region_space)
					{
						int cur_x = max(min(int(lt.x/down_scale+j+x+0.5),(imgwidth-1)/down_scale),0);
						int cur_y = max(min(int(lt.y/down_scale+i+y+0.5),(imgheight-1)/down_scale),0);
						//Hist_2_CumuHist(current_hist_trans, new_cumu_hist, Point(cur_x, cur_y), imgwidth/down_scale);
						//Hist_2_CumuHist(template_hist_trans, template_cumu_hist, Point(int(selection.x/down_scale+x+0.5),int(selection.y/down_scale+y+0.5)),imgwidth/down_scale);
						//distance_storage[k] = Histogram_distance_EMD(new_cumu_hist, template_cumu_hist);
						distance_storage[k] = Histogram_distance_naive(current_hist_trans, template_hist_trans, Point(cur_x, cur_y), Point(int(selection.x/down_scale+x+0.5),int(selection.y/down_scale+y+0.5)), imgwidth/down_scale);
						accuvalue += distance_storage[k];
						k++;
					}
				}
				counterx=k;
				for(int n = 0;n<k;n++)
					temp_distance_storage[n] = distance_storage[n];
				new_dist = randomselect(temp_distance_storage,0,k-1,k/detect_per);
				if(new_dist < lowest_dist)
				{
					lowest_dist = new_dist;
					new_lt = Point(lt.x+j*down_scale,lt.y+i*down_scale);
					//new_center = Point(current_center.x+j*down_scale,current_center.y+i*down_scale);
					meanvalue = accuvalue/k;
					for(int n = 0;n<k;n++)
						best_distance_storage[n] = distance_storage[n];
				}
			}
		}

		new_lt = Point(min(max(new_lt.x,0),int(imgwidth-selection.width/2+0.5)),min(max(new_lt.y,0),int(imgheight-selection.height/2+0.5)));
		new_center = new_lt + Point(selection.width/2, selection.height/2);
		/**********************Tracking****************************/
		t = (double)cvGetTickCount() - t;

		printf( "%.5f second, %.5f fps\n", t/(cvGetTickFrequency()*1000000.),1/(t/(cvGetTickFrequency()*1000000.)));
		avgFPS += 1/(t/(cvGetTickFrequency()*1000000.));num_f++;

		printf("region_space = %f, sigma = %f\n",region_space,LSH_Spatial_sigma);
		printf("mean=%f\n",meanvalue);
		//printf("distance=%f\n",new_dist);
		//printf("%d,%d,%d,%d,%d\n",counterx,region_num_x,region_num_y,selection.width,selection.height);
		current_center = new_center;
		lt = new_lt;
		int counting_n = 0;int update_counting = 0;

		outfile<<lt.x<<' '<<lt.y<<' '<<selection.width<<' '<<selection.height<<' '<<endl;

		/**********************Draw all the pixels****************************/
		for(float y=region_space/2;y<selection.height-region_space/2;y+=region_space)
		{
			for(float x=region_space/2;x<selection.width-region_space/2;x+=region_space)
			{
				circle(img, lt+Point(int(x+0.5),int(y+0.5)),1, CV_RGB(lowest_dist*255*10,0,0),1);
			}
		}
		/**********************Draw all the pixels****************************/

		/**********************Update template and draw the updated the regions****************************/
		for(float y=region_space/2;y<selection.height/down_scale-region_space/2;y+=region_space)
		{
			for(float x=region_space/2;x<selection.width/down_scale-region_space/2;x+=region_space)
			{
				int cur_x = max(min(int(lt.x/down_scale+x+0.5),(imgwidth-1)/down_scale),0);
				int cur_y = max(min(int(lt.y/down_scale+y+0.5),(imgheight-1)/down_scale),0);

				if(best_distance_storage[counting_n] > meanvalue*(1.0-ff) && best_distance_storage[counting_n] < meanvalue*(1.0+ff))
				{
					update_counting++;
					for(int k = 0;k<hist_bin_size;k++)
						template_hist_trans[int(selection.x/down_scale+x+0.5)+(int(selection.y/down_scale+y+0.5))*imgwidth/down_scale][k]
					= current_hist_trans[cur_x + cur_y*imgwidth/down_scale][k];
					circle(img, lt+Point(int(x*down_scale+0.5),int(y*down_scale+0.5)),1, CV_RGB(0,0,255),1);
				}
				counting_n++;
			}
		}
		/**********************Update template and draw the updated the regions****************************/

		printf("total # of regions=%d,updated regions=%d\n",counterx,update_counting);

		/**********************Draw the tracking object****************************/
		circle(img, new_center, 3, CV_RGB(lowest_dist*255*10,0,0),3);
		rectangle(img, lt, lt+Point(selection.width,selection.height),CV_RGB(lowest_dist*255*10,0,0),3);
		Mat current_tracking;
		getRectSubPix(frame, Size(selection.width,selection.height), new_center, current_tracking);
		if(visualize)
			imshow("Current_Tracking",current_tracking);
		getRectSubPix(grayimg_scale, Size(selection.width/down_scale,selection.height/down_scale), new_center*(1.0/down_scale), current_tracking);
		if(visualize)
			imshow("Current_Tracking_Feature",current_tracking);
		//printf("new_center=(%d,%d)\n",new_center.x,new_center.y);
		//printf("new=%f,old=%f\n",current_dist,new_dist);
		if(visualize)
			imshow( "Tracking", img );		
		if(visualize)
			imshow( "Feature", grayimg_scale );	
		/**********************Draw the tracking object****************************/
		
		char c = cvWaitKey(33);
		if( c == 27 ) break;
	}
	outFPS<<avgFPS/num_f<<endl;
	outfile.close();
	outFPS.close();
	he_freed(iif_current_hist);he_freed(current_hist);he_freed(current_hist_trans);he_freed(histogram_temp);he_freed(temp3w);he_freed(template_hist);he_freed(template_hist_trans);
	delete[] hist_weight, hist_weight=0;
	delete[] template_cumu_hist, template_cumu_hist=0;
	delete[] new_cumu_hist, new_cumu_hist=0;
	delete[] gaussian, gaussian=0;
	delete[] distance_storage, distance_storage=0;
	delete[] best_distance_storage, best_distance_storage=0;
	delete[] temp_distance_storage, temp_distance_storage=0;
	destroyAllWindows();
	return 0;
}