cmake_minimum_required(VERSION 3.21)
project(Camera_Based_Manipulator_TFLite)

set(CMAKE_CXX_STANDARD 14)

set(FP16_SOURCE_DIR "$ENV{HOME}/TFLite/build/FP16-source")
set(FXDIV_SOURCE_DIR  "$ENV{HOME}/TFLite/build/FXdiv-source")
set(PTHREADPOOL_SOURCE_DIR "$ENV{HOME}/TFLite/build/pthreadpool-source")
set(PSIMD_SOURCE_DIR "$ENV{HOME}/TFLite/build/psimd-source")
set(TENSORFLOW_SOURCE_DIR "$ENV{HOME}/TFLite/src" CACHE PATH
        "$ENV{HOME}/CLionProjects/camera-based-manipulator-tflite")
if(NOT TENSORFLOW_SOURCE_DIR)
    get_filename_component(TENSORFLOW_SOURCE_DIR
            "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
        "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
        "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(Camera_Based_Manipulator_TFLite main.cpp face_detector.h face_detector.cpp)

#OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

#TFLite
target_link_libraries(${PROJECT_NAME} tensorflow-lite)