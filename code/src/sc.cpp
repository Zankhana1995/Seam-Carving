
//============================================================================
// Name        : sc.cpp
// Author      : Zankhanaben Patel
// Version     :
// Copyright   : Your copyright notice
//============================================================================


#include "sc.h"

using namespace cv;
using namespace std;

int max_val = 10;
vector<int> vec;
int init_Val = 1;
int seam = 100;
vector<int> seam_vector;
int zero = 0;


//Entry point for the file.
bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image){
    
    // some sanity checks
    // Check 1 -> new_width <= in_image.cols
    if(new_width>in_image.cols){
        cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_height>in_image.rows){
        cout<<"Invalid request!!! ne_height has to be smaller than the current size!"<<endl;
        return false;
    }
    
    if(new_width<=0){
        cout<<"Invalid request!!! new_width has to be positive!"<<endl;
        return false;
        
    }
    
    if(new_height<=0){
        cout<<"Invalid request!!! new_height has to be positive!"<<endl;
        return false;
        
    }
    
    
    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}


//this method will fill pixel_energy_source vector by using various condition.
void initEnergyVector(int rows, int cols, Mat &gradient_energy, vector< vector< int > >& pixel_energy_Source){
    
    
    for(int col_index = zero; col_index < cols; col_index++){
        pixel_energy_Source[zero][col_index]= (int)gradient_energy.at<uchar>(zero,col_index);
    }
    
    
    for(int row = init_Val; row < rows; row++){
        for(int col = zero; col < cols; col++){
            
            if (col == zero){
                pixel_energy_Source[row][col] = min(pixel_energy_Source[row-init_Val][col+init_Val], pixel_energy_Source[row-init_Val][col]);
            }
            
            else if (col == cols-init_Val){
                pixel_energy_Source[row][col] = min(pixel_energy_Source[row-init_Val][col-init_Val], pixel_energy_Source[row-init_Val][col]);
            }
            
            else{
                
                pixel_energy_Source[row][col] = min(min(pixel_energy_Source[row-init_Val][col-init_Val], pixel_energy_Source[row-init_Val][col]), pixel_energy_Source[row-init_Val][col+init_Val]);
            }
            
            
            pixel_energy_Source[row][col] += (int)gradient_energy.at<uchar>(row,col);
        }
    }
    
    for(int index_x = init_Val; index_x < max_val; index_x++ ){
        vec.push_back(index_x);
    }
    
    
}

//find min col value form energy vector 
int findMinimumColumnValues(int rows, int cols, vector< vector< int > >& pixel_energy_Source){
    
    int minimum_value_column = zero;
    int minimum_Value = pixel_energy_Source[rows-init_Val][zero];
    
    
    for(int col_index = init_Val;col_index < cols; col_index++){
        if(minimum_Value > pixel_energy_Source[rows-init_Val][col_index]){
            minimum_Value = pixel_energy_Source[rows-init_Val][col_index];
            minimum_value_column = col_index;
        }
    }
    return minimum_value_column;
}

//this method find perfect seam as per input image and col values and final result will store in out img.
void findSeamCarving(int rows, int cols, vector< vector< int > >& pixel_energy_Source, Mat &gradient_energy, vector< int >& finalSeam_vec){
    
    
    int row = rows-init_Val;
    
    int minColVal = findMinimumColumnValues(rows,cols,pixel_energy_Source);
    
    finalSeam_vec[row] = minColVal;
    
    while(row != zero){
        
        int val = pixel_energy_Source[row][minColVal] - (int)gradient_energy.at<uchar>(row,minColVal);
        
        if(minColVal == cols-init_Val){
            if(val == pixel_energy_Source[row-init_Val][minColVal-init_Val])
                minColVal = minColVal - init_Val;
        }
        
        else if(minColVal == zero){
            
            if(val == pixel_energy_Source[row-init_Val][minColVal+init_Val])
                minColVal = minColVal+init_Val;
        }
        
        else{
            
            if(val == pixel_energy_Source[row-init_Val][minColVal-init_Val])
                minColVal = minColVal-init_Val;
            else if(val == pixel_energy_Source[row-init_Val][minColVal+init_Val])
                minColVal=minColVal+init_Val;
        }
        
        row--;
        finalSeam_vec[row]=minColVal;
    }
    
    
}

//get final updated result 
void getFinalModifiedImage(Mat& iimage, int rows, int cols, Mat& iiiImage, vector< int >& finalSeam_vec ,Mat& temp_Img){
    
    
    int rowSeam = iimage.rows;
    int colSeam= iimage.cols;
    
    for(int rowIndex = zero; rowIndex < rowSeam; rowIndex++){
        
        int count = zero;
        
        for(int colIndex = zero; colIndex < colSeam; colIndex++){
            
            if(colIndex != finalSeam_vec[rowIndex]){
                
                iiiImage.at<Vec3b>(rowIndex,count) = iimage.at<Vec3b>(rowIndex,colIndex);
                temp_Img.at<Vec3b>(rowIndex,count) = iimage.at<Vec3b>(rowIndex,colIndex);
                
                count++;
            }
        }
    }
    
}


//Entry point for executing method 
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image){
    
    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();
    
    Mat gaussianBlur_Out = in_image.clone();
    Mat gaussianBlur_Img_temp = in_image.clone();
    
    
    int colsDiff = in_image.cols - new_width;
    
//GaussianBlur for smoothing the Image
    GaussianBlur(gaussianBlur_Img_temp, gaussianBlur_Img_temp,Size(3,3),0,0, BORDER_DEFAULT);
    
    int rowsDiff = in_image.rows - new_height;
    
    
    for(int cols = zero; cols < colsDiff; cols++){
        reduce_horizontal_seam_trivial(iimage,oimage,gaussianBlur_Img_temp,false);
        iimage=oimage.clone();
    }
    
    for(int rows = zero; rows < rowsDiff; rows++){
        reduce_horizontal_seam_trivial(iimage,oimage,gaussianBlur_Img_temp,true);
        iimage=oimage.clone();
    }
    
    out_image = oimage.clone();
    return true;
}

//Find horizontal seam 
bool reduce_horizontal_seam_trivial(Mat& iimage, Mat& oimage, Mat& gaussianBlur_Img_temp,bool image_check){
    
    if(image_check) {
        
        transpose(iimage, iimage);
        flip(iimage, iimage,init_Val);
        transpose(gaussianBlur_Img_temp, gaussianBlur_Img_temp);
        flip(gaussianBlur_Img_temp, gaussianBlur_Img_temp,init_Val);
        
    }
    
    Mat gaussian;
    
//convert image to one to another color space
    cvtColor(iimage,gaussian,CV_BGR2GRAY);
    
    Mat gaussian_Img_x;
    
    Mat gaussian_Img_temp_abs;
    
//for smoothing and differentiation for the calculate derivaties from imgae (kernal size = 3)
    Sobel(gaussian, gaussian_Img_x, CV_32F, init_Val, 0,3);
    
//Convert result to 8 bit.
    convertScaleAbs( gaussian_Img_x, gaussian_Img_temp_abs );
    
    gaussian_Img_x.release();
    
    Mat gaussian_Img_y;
    
//for smoothing and differentiation for the calculate derivaties from imgae (kernal size = 3)
    Sobel(gaussian, gaussian_Img_y, CV_32F, 0, init_Val,3);
    gaussian.release();
    
    Mat gaussian_Img_temp_abs_y;
    
//Convert result to 8 bit.
    convertScaleAbs( gaussian_Img_y, gaussian_Img_temp_abs_y );
    
    gaussian_Img_y.release();
    
    Mat gradient_energy;
    
//calculates the weighted 

//https://docs.opencv.org/2.4.13.7/modules/core/doc/operations_on_arrays.html#void%20addWeighted(InputArray%20src1,%20double%20alpha,%20InputArray%20src2,%20double%20beta,%20double%20gamma,%20OutputArray%20dst,%20int%20dtype)

    addWeighted( gaussian_Img_temp_abs, 0.5, gaussian_Img_temp_abs_y, 0.5, 0, gradient_energy);
    
    gaussian_Img_temp_abs.release();
    gaussian_Img_temp_abs_y.release();
    
    
    
    
    int cols = gradient_energy.cols;
    int rows = gradient_energy.rows;
    
    
    vector<vector<int> > pixel_energy_Source;
    pixel_energy_Source.resize(rows, std::vector<int>(cols, 0));
    
    
    initEnergyVector(rows, cols, gradient_energy, pixel_energy_Source);
    
    
    vector<int> finalSeam_vec(rows);
    findSeamCarving(rows,cols,pixel_energy_Source,gradient_energy,finalSeam_vec);
    
    
    
    
    Mat iiiImage = Mat(iimage.rows, iimage.cols-init_Val, CV_8UC3);
    Mat temp_Img = Mat(gaussianBlur_Img_temp.rows, gaussianBlur_Img_temp.cols-init_Val, CV_8UC3);
    
    
    
    getFinalModifiedImage(iimage, rows, cols, iiiImage, finalSeam_vec ,temp_Img);
    
    
    
    oimage = iiiImage.clone();
    iiiImage.release();
    gaussianBlur_Img_temp = temp_Img.clone();
    temp_Img.release();
    
    if(image_check){
        
        
        transpose(oimage, oimage);
        flip(oimage, oimage,0);
        
        transpose(gaussianBlur_Img_temp, gaussianBlur_Img_temp);
        flip(gaussianBlur_Img_temp, gaussianBlur_Img_temp,0);
        
        
        
        
        
    }
    
    return true;
}


//Find vertical seam 
bool reduce_vertical_seam_trivial(Mat& iimage, Mat& oimage, Mat& gaussianBlur_Img_temp,bool image_check){
    
    if(image_check) {
        
        transpose(iimage, iimage);
        flip(iimage, iimage,init_Val);
        transpose(gaussianBlur_Img_temp, gaussianBlur_Img_temp);
        flip(gaussianBlur_Img_temp, gaussianBlur_Img_temp,init_Val);
        
    }
    
    Mat gaussian;
    
//convert image to one to another color space
    cvtColor(iimage,gaussian,CV_BGR2GRAY);
    
    Mat gaussian_Img_x;
    
    Mat gaussian_Img_temp_abs;
    
//for smoothing and differentiation for the calculate derivaties from imgae (kernal size = 3)
    Sobel(gaussian, gaussian_Img_x, CV_32F, init_Val, 0,3);
    
//Convert result to 8 bit.
    convertScaleAbs( gaussian_Img_x, gaussian_Img_temp_abs );
    
    gaussian_Img_x.release();
    
    Mat gaussian_Img_y;
    
//for smoothing and differentiation for the calculate derivaties from imgae (kernal size = 3)
    Sobel(gaussian, gaussian_Img_y, CV_32F, 0, init_Val,3);
    gaussian.release();
    
    Mat gaussian_Img_temp_abs_y;
    
//Convert result to 8 bit.
    convertScaleAbs( gaussian_Img_y, gaussian_Img_temp_abs_y );
    
    gaussian_Img_y.release();
    
    Mat gradient_energy;
    
//calculates the weighted 

//https://docs.opencv.org/2.4.13.7/modules/core/doc/operations_on_arrays.html#void%20addWeighted(InputArray%20src1,%20double%20alpha,%20InputArray%20src2,%20double%20beta,%20double%20gamma,%20OutputArray%20dst,%20int%20dtype)

    addWeighted( gaussian_Img_temp_abs, 0.5, gaussian_Img_temp_abs_y, 0.5, 0, gradient_energy);
    
    gaussian_Img_temp_abs.release();
    gaussian_Img_temp_abs_y.release();
    
    
    
    
    int cols = gradient_energy.cols;
    int rows = gradient_energy.rows;
    
    
    vector<vector<int> > pixel_energy_Source;
    pixel_energy_Source.resize(rows, std::vector<int>(cols, 0));
    
    
    initEnergyVector(rows, cols, gradient_energy, pixel_energy_Source);
    
    
    vector<int> finalSeam_vec(rows);
    findSeamCarving(rows,cols,pixel_energy_Source,gradient_energy,finalSeam_vec);
    
    
    
    
    Mat iiiImage = Mat(iimage.rows, iimage.cols-init_Val, CV_8UC3);
    Mat temp_Img = Mat(gaussianBlur_Img_temp.rows, gaussianBlur_Img_temp.cols-init_Val, CV_8UC3);
    
    
    
    getFinalModifiedImage(iimage, rows, cols, iiiImage, finalSeam_vec ,temp_Img);
    
    
    
    oimage = iiiImage.clone();
    iiiImage.release();
    gaussianBlur_Img_temp = temp_Img.clone();
    temp_Img.release();
    
    if(image_check){
        
        
        transpose(oimage, oimage);
        flip(oimage, oimage,0);
        
        transpose(gaussianBlur_Img_temp, gaussianBlur_Img_temp);
        flip(gaussianBlur_Img_temp, gaussianBlur_Img_temp,0);
        
        
        
        
        
    }
    
    return true;
}

