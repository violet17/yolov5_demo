# YOLOv5_OpenVINO_demo

YOLOv5 model for OpenVINO inference

##	Convert Weights to ONNX File
The following components are required.
-	OpenVINO - in this document we validate with OpenVINO linux version 2021.1 release
-	System – CPU processor. This guide is validated with Sky Lake on Linux Ubuntu 18.04.1
-	Python – in this document we validate with Python 3.6.9
-	ONNX – in this document we validate with ONNX 1.6.0
-	Pytorch – in this document we validate with Pytorch 1.6.0
-	Netron – in this document we validate with Netron 4.4.3.

###	Clone YOLOv5 Repository from Github
Running the following command in the terminal of Linux (the commit 4d3680c is used to validate in this document). 

```
$ git clone https://github.com/ultralytics/yolov5
```


###	Set up the Environment of YOLOv5
To set up the environment of YOLOv5, we need to install some requirements by running the following command:

```
$ pip install -r requirements.txt
$ pip install onnx
```

###	Download Pytorch Weights 
There are three tag in YOLOv5 repository so far. And YOLOv5 includes YOLOv5s, YOLOv5m, YOLOv5l and YOLOv5x due to different backbone. Here we use YOLOv5s from tag v3.0 for description. Run the following command to download yolov5s.pt:

```
$ wget https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt 
```


###	Convert Pytorch Weights to ONNX Weights 
The YOLOv5 repository provides a script models/export.py to export Pytorch weights with extensions *.pt to ONNX weights with extensions *.onnx.

Since OpenVINO 2021.1 hasn’t fully support ONNX opset version 11, we need to revise the script models/export.py in line 69 to opset version 10: 

```python
torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['images'],
                    output_names=['classes', 'boxes'] if y is None else ['output'])
```


Then, we save the script and run the following command:

```
$ python models/export.py  --weights yolov5-v3/yolov5s.pt  --img 640 --batch 1
```

Then we can get yolov5s.onnx in yolov5-v3 folder.



##	Convert ONNX File to IR File
After we get ONNX weights file from the last section, we can convert it to IR file with model optimizer. 
Run the following script to temporarily set OpenVINO environment and variables:

```
$ source /opt/intel/openvino_2021/bin/setupvars.sh
```

We need to specify the output node of the IR when we use model optimizer to convert the YOLOv5 model.

There are 3 output nodes in YOLOv5. We can use Netron to visualize the YOLOv5 ONNX weights. Then we find the output nodes by searching the keyword “Transpose” in Netron. After that, we can find the convolution node marked as oval shown in following Figure. After double clicking the convolution node, we can read its name “Conv_455” for stride 8 on the properties panel marked as rectangle shown in following Figure.  The following Figure shows the output node with size 1x3x80x80x85 for the input image with resolution 1x3x640x640, which is used to detect small objects. We apply this name “Conv_455” of convolution node to specify the model optimizer parameters.

<img src="https://github.com/violet17/yolov5_demo/blob/main/yolov5_output_node_for_stride_8.png" width="70%">

Similarly, we can find the other two output nodes “Conv_471” for stride 16 and “Conv_487” for stride 32.

we can run the following command to generate the IR of YOLOv5 model:
```
$ python /opt/intel/openvino_2021.1.110/deployment_tools/model_optimizer/mo.py  --input_model yolov5-v3/yolov5s.onnx --model_name yolov5-v3/yolov5s -s 255 --reverse_input_channels --output Conv_487,Conv_471,Conv_455
```

Where --input_model defines the pre-trained model,  the parameter --model_name is name of the network in generated IR and output .xml/.bin files, -s represents that all input values coming from original network inputs will be divided by this value, --reverse_input_channels is used to switch the input channels order from RGB to BGR (or vice versa), --output represents the name of the output operation of the model.

After that, we can get IR of YOLOv5s in FP32 in folder yolov5-v3.

##	OpenVINO Inference Python Demo
After generate IR of YOLOv5 model, we write the inference Python demo according to the inference process of YOLOv5 model. Based on the YOLOv3 demo provided in OpenVINO default Python demos, there are mainly three points need to be revised in YOLOv5 demo:
-	Preprocessing input images by letterbox
-	YOLO region layer using Sigmoid function
-	Post-processing of the bounding box

The inference Python demo can be found at [yolov5_demo.py](https://github.com/violet17/yolov5_demo/blob/main/yolov5_demo.py)

## Reference

OpenVINO[https://github.com/openvinotoolkit/openvino]

YOLOv5[https://github.com/ultralytics/yolov5]
