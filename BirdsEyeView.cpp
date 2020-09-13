
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
#define PI 3.1415926


int frameWidth = 800;
int frameHeight = 450;

float toRads(int degs) {
    return degs * (PI/180);
}

void transform_t(cv::Point2i &out_pt, const cv::Point2i &in_pt, const cv::Mat &HomographyMat_3x3) {

    const float *m = HomographyMat_3x3.ptr<float>();
    const float scale = m[6]*in_pt.x + m[7]*in_pt.y + m[8];

    out_pt.x = (m[0]*in_pt.x + m[1]*in_pt.y + m[2]) / scale;
    out_pt.y = (m[3]*in_pt.x + m[4]*in_pt.y + m[5]) / scale;
}

cv::Mat get3dAffineMatrix(float roll, float pitch, float yaw, float z_dist) {

    const float sin_x = sin(-roll);
    const float cos_x = cos(-roll);

    const float sin_y = sin(-pitch);
    const float cos_y = cos(-pitch);

    const float sin_z = sin(yaw);
    const float cos_z = cos(yaw);

    //Original
    return (Mat_<float>(4, 4) <<
             (cos_y * cos_z),  (sin_x * sin_y * cos_z - cos_x * sin_z),  (cos_x * sin_y * cos_z + sin_x * sin_z),   0,
             (cos_y * sin_z),  (sin_x * sin_y * sin_z + cos_x * cos_z),  (cos_x * sin_y * sin_z - sin_x * cos_z),   0,
             (-sin_y),         (sin_x * cos_y),                          (cos_x * cos_y),                           z_dist,
             0, 0, 0, 1);
}

cv::Mat getPaddedTranslationMatrix(cv::Mat& projectiveMat, int w, int h) {

    Mat Tm_inv;
    invert(projectiveMat , Tm_inv, DECOMP_LU);
    Mat T1 = Tm_inv * (Mat_<float>(3, 1) << (float)w, (float)h, 1); 
    float t1x = T1.at<float>(0,0) / T1.at<float>(2,0);
    float t1y = T1.at<float>(1,0) / T1.at<float>(2,0);

    Mat T2 = Tm_inv * (Mat_<float>(3, 1) << 0, (float)h, 1); 
    float t2x = T2.at<float>(0,0) / T2.at<float>(2,0);
    float t2y = T2.at<float>(1,0) / T2.at<float>(2,0);

    return (Mat_<float>(4, 3)<< 
        1, 0, -w/2,
        0, 1, max(t1y - h, t2y - h),
        0, 0, 0,
        0, 0, 1);
}

int main() {

    string filename = "examples/video2.mp4"; // File name
    VideoCapture capture(filename);
  
    Mat source, source1, destination;

    // Initialisation of parameters
    int alpha_ = 145, beta_ = 90, gamma_ = 90; 
    int f_ = 500, dist_ = 200;
    int pxstep_ = 150;
    int image_h = 250;
    int image_w = 900;

    Rect Rec(1280 - 200 - image_w, 720 - image_h - 30, image_w, image_h); //ROI.

    // Initialisation of window
    namedWindow("Original", 1);
    namedWindow("Result", 1);
    createTrackbar("Row", "Result", &alpha_, 180);
    createTrackbar("Pitch", "Result", &beta_, 180);
    createTrackbar("Yaw", "Result", &gamma_, 180);
    createTrackbar("Focal", "Result", &f_, 2000);
    createTrackbar("Distance", "Result", &dist_, 2000);


    while( true ) {
        
        capture >> source1;
        rectangle(source1, Rec, Scalar(255), 1, 8, 0); //ROI
        source = source1(Rec);
        imshow("Original", source1);

        resize(source, source,Size(frameWidth, frameHeight));

        Size image_size = source.size();
        double w = (double)image_size.width, h = (double)image_size.height;
        
        // matrix projection 2D to 3D
        Mat A1 = (Mat_<float>(4, 3)<< 
            1, 0, 0,
            0, 1, 0,
            0, 0, 0,
            0, 0, 1 );
            
        // RT - affine matrix
        Mat RT = get3dAffineMatrix(toRads(alpha_ -90), toRads(beta_ -90), toRads(gamma_ -90), dist_);

        // K - camera parameter matrix 
        Mat K = (Mat_<float>(3, 4) << 
            f_, 0, w/2, 0,
            0, f_, h/2, 0,
            0, 0, 1, 0); 

        // calculate the padding offset for output image
        Mat Tm = K * RT * A1;
        A1 = getPaddedTranslationMatrix(Tm, w, h);

        // wrap perspective
        Mat transformationMat = K * RT * A1;
        warpPerspective(source, destination, transformationMat, image_size, INTER_CUBIC | WARP_INVERSE_MAP);
        imshow("Result", destination);

        //Pause
        if (cv::waitKey(50) == 'p')
            while (cv::waitKey(5) != 'p');

    }


    return 0;
}
