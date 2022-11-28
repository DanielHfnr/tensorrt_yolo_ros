C++ ROS node for TensorRT inference of YoloV7 (yolov7-tiny) object detection. Node load a yolov7.onnx model and create a TensorRT engine from that. Gets camera input data, runs inference and postprocesses network outputs. 

Publishes detected bounding boxes as well as rendered input image with bounding box overlay. 

**Tested with:**

- Ubuntu 20.04
- Nvidia Driver 520
- CUDA 11.6
- TensorRT 8.4.3
- ROS Noetic

### Prerequisites 

- Installed ROS: http://wiki.ros.org/noetic/Installation/Ubuntu
- Install CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
- Install TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

Follow the instructions to install ROS, CUDA and TensorRT.

Install ROS vision messages:

```
sudo apt-get install ros-distro-vision-msgs
```

**Optional:**

Install clang-format for code formatting: 
```
sudo apt-get install clang-format
```

### Usage

Clone the repository into your catkin workspace src folder:

```
git clone https://github.com/DanielHfnr/tensorrt_yolo_ros.git
```

Build you workspace using `catkin_make` or `catkin build`.

Launch the ROS node using the provided launch file (adjust topic names where needed):

```
roslaunch tensorrt_yolo tensorrt_yolo.launch
```

```
<launch>
  <arg name="output" default="screen"/>
  
  <node name="tensorrt_yolo" pkg="tensorrt_yolo" type="tensorrt_yolo_node" output="$(arg output)">
      <param name="yolo_onnx_model" value="$(find tensorrt_yolo)/models/yolov7-tiny.onnx" type="str" />
      <param name="class_labels_file" value="$(find tensorrt_yolo)/models/coco_classes.txt" type="str" />

      <param name="device" value="GPU" type="str" />
      <param name="precision" value="FP32" type="str" />
      <param name="allow_gpu_fallback" value="true" type="bool" />


      <param name="image_topic_in" value="/camera/image_raw" type="str" />
      <param name="image_topic_out" value="/camera/image_detections" type="str" />
      <param name="bounding_boxes_topic_out" value="/object_detection/bounding_boxes_2d" type="str" />

      <param name="show_output_image" value="true" type="bool" />
  </node>
</launch>
```


To test if everything works correctly get camera input data. For that you can for examples use the following ros package: [http://wiki.ros.org/video_stream_opencv](http://wiki.ros.org/video_stream_opencv)

