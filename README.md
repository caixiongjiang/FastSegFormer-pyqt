## FastSegFormer-pyqt

[中文](https://github.com/caixiongjiang/FastSegFormer-pyqt/blob/main/README_CH.md)

Navel orange defect segmentation model video detection UI.

### Update

- [x] Create PyQT interface for navel orange defect segmentation. (May/10/2023)
- [x] Produce 30 frames of navel orange assembly line simulation video. (May/13/2023)
- [x] Support onnx format for video detection. (May/14/2023)
- [x] Using multi-threaded processing, the main thread updates the UI and the sub-threads are used to process the video frames and improve the FPS to 48~60.(May/25/2023)
- [x] Deployed on Jetson Nano(4G), an edge computing device with ONNXRuntime and TensorRT.(May/30/2023)
- [ ] Acceleration with DeepStream Framework on Jetson Nano(4G).

### Demo

<p align="center">
  <img src="Figs/orange_video.gif" alt="Cityscapes" width="360"/>
  <img src="Figs/orange_detection_video.gif" alt="Cityscapes" width="360"/></br>
  <span align="center">Navel orange simulation line detection video</span>
</p>


### Usage

* Environment Configuration：
```shell
$ conda activate 'your anaconda environment'
$ pip install -r requirements.txt 
```
* Run project:
```shell
python run_gui.py
```
* Jetson Nano Deployment: [Usage](https://github.com/caixiongjiang/FastSegFormer-pyqt/tree/main/jetson-FastSegFormer)

### Testing performance comparison

> All the following tests are not only network inference, but also include pre-processing and post-processing, and the processing of video frames may be different for different methods.

* System: Windows 10, 
  CPU: Intel(R) Core(TM) i5-10500 CPU @ 3.10GHz
  GPU: NVIDIA GeForce RTX 3060(12G)

<table>
	<tr>
	    <th colspan="8">FastSegFormer-pyqt</th>
	</tr >
	<tr>
	    <td style="text-align: center;">Task</td>
	    <td style="text-align: center;">Video input</td>
	    <td style="text-align: center;">Inference input</td>  
      <td style="text-align: center;">Inference framework</td>
      <td style="text-align: center;">GPU computing capability</td>
      <td style="text-align: center;">Quantification</td>
      <td style="text-align: center;">Video processing</td>
      <td style="text-align: center;">Average FPS</td>
	</tr >
	<tr >
	    <td rowspan="6" style="text-align: center;">Video Detection</td>
	    <td rowspan="6" style="text-align: center;">$512\times 512$</td>
	    <td rowspan="6" style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;">PyTorch</td>
      <td rowspan="6" style="text-align: center;">12.74 TFLOPS</td>
      <td rowspan="2" style="text-align: center;">FP32</td>
      <td rowspan="4" style="text-align: center;">Single frame</td>
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
      <td rowspan="2" style="text-align: center;">Multi-thread</td>
      <td style="text-align: center;">46.94</td>
	</tr>
  <tr>
	    <td style="text-align: center;">ONNXRuntime</td>
      <td style="text-align: center;">46.81</td>
	</tr>
</table>


Conclusion：
1. In the mode of GPU for inference, there is almost no difference in inference time between PyTorch and ONNXRuntime.
2. There is almost no difference in GPU inference time using cards with the same FP32 and FP16 arithmetic power.
3. The main thread handles the input and output of the video, and the secondary thread handles the inference of the single-frame image, which has a great improvement on the detection performance of the video.


* System: Ubuntu 18.04
  CPU: ARM Cortex-A57 @ 1.43GHz
  GPU: NVIDIA Maxwell @ 921MHz

<table>
	<tr>
	    <th colspan="8">Jetson-FastSegFormer</th>
	</tr >
	<tr>
	    <td style="text-align: center;">Task</td>
	    <td style="text-align: center;">Video/Stream input</td>
	    <td style="text-align: center;">Inference input</td>  
      <td style="text-align: center;">Inference framework</td>
      <td style="text-align: center;">GPU computing capability</td>
      <td style="text-align: center;">Quantification</td>
      <td style="text-align: center;">Video processing</td>
      <td style="text-align: center;">Average FPS</td>
	</tr >
	<tr >
	    <td rowspan="5" style="text-align: center;">Video Detection</td>
	    <td rowspan="5" style="text-align: center;">$512\times 512$</td>
	    <td rowspan="10" style="text-align: center;">$224\times 224$</td>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="10" style="text-align: center;"> 0.4716 TFLOPS</td>
      <td rowspan="10" style="text-align: center;">FP16</td>
      <td rowspan="2" style="text-align: center;">Single frame</td>
      <td style="text-align: center;">7.02</td>
	</tr>
	<tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">11.50</td>
	</tr>
	<tr>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">Multi-thread</td>
      <td style="text-align: center;">~</td>
	</tr>
	<tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
	    <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">DeepStream</td>
      <td style="text-align: center;">--</td>
	</tr>
  <tr>
	    <td rowspan="5" style="text-align: center;">CSI Camera Detection</td>
      <td rowspan="5" style="text-align: center;">$640\times 480$</td>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">Single frame</td>
      <td style="text-align: center;">5</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">7</td>
	</tr>
  <tr>
      <td style="text-align: center;">ONNXRuntime</td>
      <td rowspan="2" style="text-align: center;">Multi-thread</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">~</td>
	</tr>
  <tr>
      <td style="text-align: center;">TensorRT</td>
      <td style="text-align: center;">DeepStream</td>
      <td style="text-align: center;">--</td>
	</tr>
</table>
~:Can't run dual-threaded acceleration on Jetson nano (4G) because of lack of memory.

--:Currently not achieved.




