## Jetson-FastSegFormer

[中文](https://github.com/caixiongjiang/FastSegFormer-pyqt/blob/main/jetson-FastSegFormer/README.md)

### Environment

* Hardware:Jetson Nano(4G) B01, CSI camera

* Package：

> deepstream-app version 6.0.1
> DeepStreamSDK 6.0.1
> CUDA Driver Version: 10.2
> CUDA Runtime Version: 10.2
> TensorRT Version: 8.2
> cuDNN Version: 8.2
> ONNXRuntime-gpu: 1.6.0 (Need to choose your own according to the hardware)
> Python: 3.6.9
> jetpack: 4.6

* Questions about ONNXRuntime-gpu version selection and installation:
    * Jetson Nano is an arm64 architecture, ONNXRuntime-gpu can not be downloaded directly through `pip`, you need to compile it yourself.NVIDIA official and ONNXRuntime official have compiled it for us, download link：[https://elinux.org/Jetson_Zoo#ONNX_Runtime](https://elinux.org/Jetson_Zoo#ONNX_Runtime)

    * Once downloaded, place the file in the current directory and execute the `pip` command:
    ```shell
    $ pip3 install https://nvidia.box.com/shared/static/49fzcqa1g4oblwxr3ikmuvhuaprqyxb7.whl
    ```
### Usage

* PyTorch->ONNX:
```shell
$ python3 pth2onnx.py
```
* ONNX->TensorRT:
```shell
$ python3 onnx2trt.py
```
If you can‘t serialize successfully, some operations of your model file (e.g. bilinear interpolation) are not compatible with the current TensorRT version or with the version of the operator used in the Pytorch to ONNX conversion. **You can also build the TensorRT engine online from an ONNX file, which allows you to skip the deserialization step.**

Task:

* Real-time video segmentation using ONNXRuntime and OpenCV:
```shell
$ python3 detect.py --mode=video --weight_type=.onnx 
```
* Real-time visualize video segmentation using ONNXRuntime and OpenCV:
```shell
$ python3 detect.py --mode=video --weight_type=.onnx --view=True
```
* Real-time video segmentation using TensorRT and OpenCV:
```shell
$ python3 detect.py --mode=video --weight_type=.trt
```
* Real-time visualize video segmentation using TensorRT and OpenCV:
```shell
$ python3 detect.py --mode=video --weight_type=.trt --view=True
```
* Real-time CSI camera segmentation using ONNXRuntime and OpenCV:
```shell
$ python3 detect.py --mode=camera --weight_type=.onnx 
```
* Real-time CSI camera segmentation using TensorRT and OpenCV:
```shell
$ python3 detect.py --mode=camera --weight_type=.trt 
```

If your Jetson device has more than 4G memory, you can use multiple threads to speed up the detection by setting the parameter `--thread=True`.

* Real-time CSI camera/video segmentation using TensorRT and DeepStream:
There are still some problems. Please refer to the official example given for modification:[deepstream-segmentation](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-segmentation) and [deepstream-segmask](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-segmask).