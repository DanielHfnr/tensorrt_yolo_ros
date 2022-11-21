# TensorrtYoloROS

C++ ROS node for TensorRT inference of YOLO object detection.


### Prerequisites

- Tensorrt Installed CUDA + Tensorrt

Optional:
Install clang for code formatting: sudo apt-get install clang-format

Install ROS vision messages

sudo apt-get install ros-distro-vision-msgs

### Getting started

Go to the YoloV7 Github pages: https://github.com/WongKinYiu/yolov7 

Follow the instructions to download a pretrained yolov7 and convert it to ONNX. Copy the ONNX model to the tensorrt_yolo/models folder. 
