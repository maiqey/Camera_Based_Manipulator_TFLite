//
// Created by maiqy on 4/5/22.
//

#include "face_detector.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <opencv2/opencv.hpp>

Detector::Detector(int crowd_type){
    switch(crowd_type){
        case 1:
            model = tflite::FlatBufferModel::BuildFromFile("../models/face_detection_full_range.tflite", &error_reporter);
        case 2:
            model = tflite::FlatBufferModel::BuildFromFile("../models/face_detection_full_range_sparse.tflite", &error_reporter);
        case 3:
            model = tflite::FlatBufferModel::BuildFromFile("../models/face_detection_short_range.tflite", &error_reporter);
        default:
            model = tflite::FlatBufferModel::BuildFromFile("../models/face_detection_short_range.tflite", &error_reporter);
    }
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(1);
}

void Detector::pinpoint_faces(cv::Mat frame){
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];
    frame.convertTo(frame, CV_32FC3);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    cv::Point3_<float>* pixel = frame.ptr<cv::Point3_<float>>(0,0);
    const cv::Point3_<float>* endPixel = pixel + frame.cols * frame.rows;
    for (; pixel != endPixel; pixel++) {
        pixel->x = ((pixel->x / 255.0) - 0.5) * 2.0;
        pixel->y = ((pixel->y / 255.0) - 0.5) * 2.0;
        pixel->z = ((pixel->z / 255.0) - 0.5) * 2.0;
    }
    cv::Mat image;
    cv::resize(frame, image, cv::Size(width, height), cv::INTER_NEAREST);
    interpreter->AllocateTensors();
    float* inputLayer = interpreter->typed_input_tensor<float>(0);
    float* outputLayer = interpreter->typed_output_tensor<float>(0);
    float* image_ptr = image.ptr<float>(0);
    memcpy(inputLayer, image.ptr<float>(0), width * height * channels * sizeof(float));
    interpreter->Invoke();
    std::cout << outputLayer[0] << "\n";
    /*TODO
     * interpretation of results
     */
}