This is C file for Visual Tracking published in the following paper:

Visual Tracking via Locality Sensitive Histograms 
Shengfeng He, Qing-Xiong Yang, Rynson Lau, Jiang Wang, and Ming-Hsuan Yang
Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.
 
If you use the program and compare with our LSHT trackers, please cite the above paper.

OpenCV is required to compile this program.
This program is tested under VC 2008, OpenCV 2.4.3 and Windows 7.

Please run lsht.bat in the release folder:
LSH_tracking.exe Tiger2 .\\img\\ 1 365 4 jpg 32 60 68 78 0 0.033 25 0.01 1

2. Tiger2: the name of the sequence.
3. .\\img\\: the path of the sequence, the images are named with 4 digit numbers, e.g. 0001.jpg.
4. 1: the start frame of the sequence.
5. 365: the last frame of the sequence.
6. 4: the string length of the image file name.
7. jpg: the format of the images.
8. x-coordinate of the top-left corner.
9. y-coordinate of the top-left corner.
10. The width of the bounding box.
11. The height of the bounding box.
12. Feature, 0->intensity, 1->illumination invariant feature.
13. The coefficient, alpha, of illumination invariant feature.
14. Search radius.
15. Forgetting factors.
16. Show tracking result.

**The parameters used in this program is different from setting used in the paper. 
**This setting is mainly for the best tracking performance in the benchmark, especially for fast motion (large search radius). 
**Reducing the search radius, number of bins and number of regions can be faster.

The program will output the bounding box (x,y) of the top-left corner, width and height of the tracker.

For more details, please refer to our paper.