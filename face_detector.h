//
// Created by maiqy on 4/5/22.
//

#ifndef CAMERA_BASED_MANIPULATOR_TFLITE_FACE_DETECTOR_H
#define CAMERA_BASED_MANIPULATOR_TFLITE_FACE_DETECTOR_H
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <opencv2/opencv.hpp>

class Detector{
public:
    Detector(int crowd_type);
    void pinpoint_faces(cv::Mat frame);

private:
    tflite::StderrReporter error_reporter;
    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
};

#endif //CAMERA_BASED_MANIPULATOR_TFLITE_FACE_DETECTOR_H
