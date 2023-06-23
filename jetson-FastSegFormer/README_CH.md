## Jetson-FastSegFormer

[English](https://github.com/caixiongjiang/FastSegFormer-pyqt/blob/main/jetson-FastSegFormer/README.md)

### 环境

* 硬件:Jetson Nano(4G) B01, CSI摄像头

* Python包和一些环境：

> deepstream-app version 6.0.1
> DeepStreamSDK 6.0.1
> CUDA Driver Version: 10.2
> CUDA Runtime Version: 10.2
> TensorRT Version: 8.2
> cuDNN Version: 8.2
> ONNXRuntime-gpu: 1.6.0 (Need to choose your own according to the hardware)
> Python: 3.6.9
> jetpack: 4.6

* 关于ONNXRuntime-gpu版本选择和安装的问题：
    * Jetson Nano是arm64架构，ONNXRuntime-gpu不能直接通过`pip`下载，需要自己编译。NVIDIA官方和ONNXRuntime官方已经为我们编译了，下载链接：[https://elinux.org/Jetson_Zoo#ONNX_Runtime](https://elinux.org/Jetson_Zoo#ONNX_Runtime)

    * 下载后，将文件放在当前目录下，并执行`pip`命令：
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
如果你不能成功序列化，你的模型文件的一些操作（如双线性插值）与当前的TensorRT版本或在Pytorch到ONNX转换中使用的运算器版本不兼容。**你也可以从ONNX文件中在线建立TensorRT引擎，这可以让你跳过反序列化步骤。**

Task:

* 使用ONNXRuntime和OpenCV进行实时视频分割：
```shell
$ python3 detect.py --mode=video --weight_type=.onnx 
```
* 使用ONNXRuntime和OpenCV进行实时视频分割并可视化：
```shell
$ python3 detect.py --mode=video --weight_type=.onnx --view=True
```
* 使用TensorRT和OpenCV进行实时视频分割：
```shell
$ python3 detect.py --mode=video --weight_type=.trt
```
* 使用TensorRT和OpenCV进行实时视频分割并可视化：
```shell
$ python3 detect.py --mode=video --weight_type=.trt --view=True
```
* 使用ONNXRuntime和OpenCV进行实时CSI相机捕获分割：
```shell
$ python3 detect.py --mode=camera --weight_type=.onnx 
```
* 使用TensorRT和OpenCV进行实时CSI相机捕获分割：
```shell
$ python3 detect.py --mode=camera --weight_type=.trt 
```

如果你的Jetson设备有超过4G的内存，你可以通过设置参数`--thread=True`来使用多线程来加快检测速度。

* 使用TensorRT和DeepStream进行实时CSI相机/视频分割：
```python
$ python3 deepstream_detect_multiStream.py
```