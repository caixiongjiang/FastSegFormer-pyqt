## FastSegFormer-pyqt

[English](https://github.com/caixiongjiang/FastSegFormer-pyqt/blob/main/README.md)

脐橙缺陷分割模型视频检测UI。

### Update

- [x] 为脐橙缺陷实时分割实现PyQT界面。(2023.5.10)
- [x] 制作30帧的脐橙流水线模拟视频。(2023.5.13)
- [x] 支持onnx格式的视频检测。(2023.5.14)
- [x] 使用多线程处理，主线程更新用户界面，子线程用于处理视频帧，将FPS提高到48~55。(2023.5.25)
- [x] 部署在Jetson Nano(4G)边缘计算设备上，使用ONNXRuntime和TensorRT进行视觉检测。 (2023.05.30)
- [ ] 在Jetson Nano(4G)上使用DeepStream框架进行加速。

### 演示

<p align="center">
  <img src="Figs/orange_video.gif" alt="Cityscapes" width="360"/>
  <img src="Figs/orange_detection_video.gif" alt="Cityscapes" width="360"/></br>
  <span align="center">脐橙模拟线检测视频</span>
</p>


### 使用

* 环境配置：
```shell
$ conda activate 'your anaconda environment'
$ pip install -r requirements.txt 
```
* 运行项目：
```shell
python run_gui.py
```
* Jetson Nano部署: [使用](https://github.com/caixiongjiang/FastSegFormer-pyqt/blob/main/jetson-FastSegFormer/README.md)

### 测试性能比较

> 以下所有测试不仅是网络推理，还包括预处理和后处理，不同方法对视频帧的处理可能不同。

* 系统: Windows 10, 
  CPU: Intel(R) Core(TM) i5-10500 CPU @ 3.10GHz
  GPU: NVIDIA GeForce RTX 3060(12G)

<table>
	<tr>
	    <th colspan="8">FastSegFormer-pyqt</th>
	</tr >
	<tr>
	    <td style="text-align: center;">任务</td>
	    <td style="text-align: center;">视频输入尺寸</td>
	    <td style="text-align: center;">推理输入尺寸</td>  
      <td style="text-align: center;">推理框架</td>
      <td style="text-align: center;">GPU计算能力</td>
      <td style="text-align: center;">量化</td>
      <td style="text-align: center;">视频帧处理方式</td>
      <td style="text-align: center;">平均FPS</td>
	</tr >
	<tr >
	    <td rowspan="6" style="text-align: center;">视频检测</td>
	    <td rowspan="6" style="text-align: center;">$512\times 512$</td>
	    <td rowspan="6" style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;">PyTorch</td>
      <td rowspan="6" style="text-align: center;">12.74 TFLOPS</td>
      <td rowspan="2" style="text-align: center;">FP32</td>
      <td rowspan="4" style="text-align: center;">逐帧处理</td>
      <td style="text-align: center;">32.62</td>
	</tr>
	<tr>
	    <td style="text-align: center;">ONNXRuntime</td>
      <td style="text-align: center;">32.64</td>
	</tr>
	<tr>
	    <td style="text-align: center;">PyTorch</td>
      <td rowspan="4" style="text-align: center;">FP16</td>
      <td style="text-align: center;">32.24</td>
	</tr>
	<tr>
	    <td style="text-align: center;">ONNXRuntime</td>
      <td style="text-align: center;">32.66</td>
	</tr>
  <tr>
	    <td style="text-align: center;">PyTorch</td>
      <td rowspan="2" style="text-align: center;">多线程</td>
      <td style="text-align: center;">46.94</td>
	</tr>
  <tr>
	    <td style="text-align: center;">ONNXRuntime</td>
      <td style="text-align: center;">46.81</td>
	</tr>
</table>


Conclusion：
1. 在GPU进行推理的模式下，PyTorch和ONNXRuntime在推理时间上几乎没有差别。
2. 使用具有相同FP32和FP16算力的显卡上，在GPU推理时间上几乎没有差别。
3. 主线程处理视频的输入和输出，副线程处理单帧图像的推理，这对视频的检测性能有很大提高。

* 系统: Ubuntu 18.04
  CPU: ARM Cortex-A57 @ 1.43GHz
  GPU: NVIDIA Maxwell @ 921MHz

<table>
	<tr>
	    <th colspan="8">Jetson-FastSegFormer</th>
	</tr >
	<tr>
	    <td style="text-align: center;">任务</td>
	    <td style="text-align: center;">视频流输入尺寸</td>
	    <td style="text-align: center;">推理输入尺寸</td>  
      <td style="text-align: center;">推理框架</td>
      <td style="text-align: center;">GPU计算能力</td>
      <td style="text-align: center;">量化</td>
      <td style="text-align: center;">视频帧处理方式</td>
      <td style="text-align: center;">平均FPS</td>
	</tr >
	<tr >
	    <td rowspan="5" style="text-align: center;">视频检测</td>
	    <td rowspan="5" style="text-align: center;">$512\times 512$</td>
	    <td rowspan="10" style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="10" style="text-align: center;"> 0.4716 TFLOPS</td>
      <td rowspan="10" style="text-align: center;">FP16</td>
      <td rowspan="2" style="text-align: center;">逐帧处理</td>
      <td style="text-align: center;">7.02</td>
	</tr>
	<tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">11.50</td>
	</tr>
	<tr>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">多线程</td>
      <td style="text-align: center;">~</td>
	</tr>
	<tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">DeepStream视觉处理框架</td>
      <td style="text-align: center;">--</td>
	</tr>
  <tr>
	    <td rowspan="5" style="text-align: center;">CSI Camera Detection</td>
      <td rowspan="5" style="text-align: center;">$640\times 480$</td>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">逐帧处理</td>
      <td style="text-align: center;">5</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">7</td>
	</tr>
  <tr>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">多线程</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">DeepStream视觉处理框架</td>
      <td style="text-align: center;">--</td>
	</tr>
</table>
~:由于内存不足，无法在Jetson nano（4G）上运行多线程加速。

--:目前尚未实现。




