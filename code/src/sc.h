//============================================================================
// Name        : sc.h
// Author      : Zankhanaben Patel
// Version     :
// Copyright   : Your copyright notice
//============================================================================



#ifndef SEAMCARVINGCOMP665156
#define SEAMCARVINGCOMP665156

#include <opencv2/opencv.hpp>

// the function you need to implement - by defaiult it calls seam_carving_trivial

bool seam_carving(cv::Mat& in_image, int new_width, int new_height, cv::Mat& out_image);

bool seam_carving_trivial(cv::Mat& in_image, int new_width, int new_height, cv::Mat& out_image);

bool reduce_horizontal_seam_trivial(cv::Mat& iimage, cv::Mat& oimage, cv::Mat& bluredImage,bool image_check);

bool reduce_vertical_seam_trivial(cv::Mat& iimage, cv::Mat& oimage, cv::Mat& bluredImage,bool image_check);

#endif

