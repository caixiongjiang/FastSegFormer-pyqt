## FastSegFormer-pyqt

Navel orange defect segmentation model video detection UI.

### Update

- [x] Create PyQT interface for navel orange defect segmentation. (May/10/2023)
- [x] Produce 30 frames of navel orange assembly line simulation video. (May/13/2023)
- [x] Support onnx format for video detection. (May/14/2023)
- [ ] Supports target tracking.

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
