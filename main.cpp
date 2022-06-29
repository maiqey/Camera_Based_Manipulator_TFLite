#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_detector.h"

int main(){
    // Choosing and loading desired model
    std::cout << "Describe the crowd by giving a specific number (default - type #3):\n" <<
              "1. <5m away, dense\n" <<
              "2. <5m away, sparse\n" <<
              "3. <2m away\n";
    int crowd_type = 3;
    std::cin >> crowd_type;
    Detector detector(crowd_type);

    // Capturing webcam input
    cv::Mat frame;
    cv::namedWindow("Camera");
    cv::VideoCapture cap(0);
    const int esc_key = 27;
    while(true){
        cap >> frame;
        detector.pinpoint_faces(frame);
        cv::imshow("Camera", frame);
        if(cv::waitKey(5) == esc_key){
            break;
        }
    }
}
