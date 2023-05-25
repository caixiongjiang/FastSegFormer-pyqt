# pyuic5 -o gui.py untitled.ui
import cv2, sys, yaml, os, torch, time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from gui import Ui_Dialog
from models.fastsegfomer.fastsegformer import FastSegFormer
from utils.utils import *
import onnxruntime




def resize_img(img, img_size=600, value=[255, 255, 255], inter=cv2.INTER_AREA):
    old_shape = img.shape[:2]
    ratio = img_size / max(old_shape)
    new_shape = [int(s * ratio) for s in old_shape[:2]]
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=inter)
    delta_h, delta_w = img_size - new_shape[0], img_size - new_shape[1]
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), borderType=cv2.BORDER_CONSTANT,
                             value=value)
    return img


class MyForm(QDialog):
    
    # 定义一个自定义信号，用于在文本更新时发出信号
    text_update_signal = QtCore.pyqtSignal(str)
    
    def __init__(self, title, textBrowser_size):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        self.save_path = 'result'
        self.save_id = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        self.now = None
        self.model = None
        self.video_count = None
        self._timer = None
        self.out = None
        
        self.ui.textBrowser.setFontPointSize(textBrowser_size)
        self.ui.label.setText(title)
        
        self.ui.pushButton_Model.clicked.connect(self.select_model)
        self.ui.pushButton_Img.clicked.connect(self.select_image_file)
        self.ui.pushButton_ImgFolder.clicked.connect(self.select_folder_file)
        self.ui.pushButton_Video.clicked.connect(self.select_video_file)
        self.ui.pushButton_Camera.clicked.connect(self.select_camera)
        self.ui.pushButton_BegDet.clicked.connect(self.begin_detect)
        self.ui.pushButton_Exit.clicked.connect(self._exit)
        self.ui.pushButton_SavePath.clicked.connect(self.select_savepath)
        self.ui.pushButton_StopDet.clicked.connect(self.stop_detect)
        self.ui.comboBox.currentIndexChanged.connect(self.comboBox_vis)
        self.show()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = None
        self.weight_type = None
        self.info = 'red: sunburn; green: ulcer; orange: wind scarring'
        self.text_update_signal.connect(self.update_text)  # 将信号与槽函数关联
    
    def update_text(self, message):
        self.ui.textBrowser.append(message)  # 更新文本

    def read_and_show_image_from_path(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        resize_image = cv2.cvtColor(resize_img(image), cv2.COLOR_RGB2BGR)
        self.ui.label_ori.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(resize_image.data, resize_image.shape[1], resize_image.shape[0], QtGui.QImage.Format_RGB888)))
        return image
    
    def show_image_from_array(self, image, ori=False, det=False):
        # QT的setPixmap读取的为BGR格式
        resize_image = cv2.cvtColor(resize_img(image), cv2.COLOR_RGB2BGR)
        if ori:
            self.ui.label_ori.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(resize_image.data, resize_image.shape[1], resize_image.shape[0], QtGui.QImage.Format_RGB888)))
        if det:
            self.ui.label_det.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(resize_image.data, resize_image.shape[1], resize_image.shape[0], QtGui.QImage.Format_RGB888)))
    
    def show_message(self, message):
        QMessageBox.information(self, "提示", message, QMessageBox.Ok)
    
    def reset_timer(self):
        self._timer.stop()
        self._timer = None
    
    def reset_video_count(self):
        if self.video_count is not None:
            self.video_count = None
    
    def reset_det_label(self):
        self.ui.label_det.setText('')
        
    def comboBox_vis(self):
        self.ui.textBrowser.append(f'track state change to {self.ui.comboBox.currentText()}')
        self.track_init()
    
    def track_init(self):
        if self.ui.comboBox.currentText() != 'NoTrack':
            self.model.track_init(self.ui.comboBox.currentText())
    
    def select_model(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选取文件', '.', 'YAML (*.yaml)')
        if fileName != '':
            self.ui.textBrowser.append(f'load yaml form {fileName}.')
            # read cfg
            with open(fileName) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)
            # init FastSegFormer-p model
            if self.cfg['model_type'] == 'FastSegFormer_P':
                self.model = FastSegFormer(num_classes=self.cfg['num_classes'], pretrained=False, backbone='poolformer_s12', Pyramid='multiscale', cnn_branch=True).to(self.device).eval()
                if self.cfg['model_path'].endswith('pth'):
                    self.weight_type = 'pth'
                    checkpoint = torch.load(self.cfg['model_path'], map_location=self.device)
                    self.model.load_state_dict(checkpoint)
                elif self.cfg['model_path'].endswith('onnx'):
                    self.weight_type = 'onnx'
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.model = onnxruntime.InferenceSession(self.cfg['model_path'], providers=providers)
            else:
                self.ui.textBrowser.append(f'load yaml failure.')
            self.ui.textBrowser.append(f'load yaml success.')
        else:
            self.ui.textBrowser.append(f'load yaml failure.')
            self.show_message('请选择yaml配置文件.')
    
    def select_image_file(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选取文件', '.', 'JPG (*.jpg);;PNG (*.png)')
        if fileName != '':
            self.reset_det_label()
            image = self.read_and_show_image_from_path(fileName)
            self.now = image
            self.ui.textBrowser.append(f'read image form {fileName}')
        else:
            self.show_message('请选择图片文件.')

    def select_folder_file(self):
        folder = QFileDialog.getExistingDirectory(self, '选择路径', '.')
        if folder != '':
            folder_list = [os.path.join(folder, i) for i in os.listdir(folder)]
            if len(folder_list) == 0:
                self.show_message('选择的文件夹内容为空.')
            else:
                self.reset_det_label()
                self.now = folder_list
                self.read_and_show_image_from_path(folder_list[0])
                self.ui.textBrowser.append(f'read folder form {folder}')
        else:
            self.show_message('请选择图片文件夹.')
    
    def select_video_file(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选取文件', '.', 'MP4 (*.mp4)')
        cap = cv2.VideoCapture(fileName)
        
        if self._timer is not None:
            self.reset_timer()
        
        if not cap.isOpened():
            self.show_message('MP4视频打开失败!')
        else:
            self.reset_det_label()
            flag, image = cap.read()
            self.show_image_from_array(image, ori=True)
            self.now = cap
            self.video_count = int(self.now.get(cv2.CAP_PROP_FRAME_COUNT))
            self.print_id = 1

    def select_camera(self):
        cap = cv2.VideoCapture(0)
        
        if self._timer is not None:
            self.reset_timer()
        
        if not cap.isOpened():
            self.show_message('视频打开失败.')
        else:
            self.reset_det_label()
            flag, image = cap.read()
            self.show_image_from_array(image, ori=True)
            self.now = cap
            self.print_id = 1
    
    def begin_detect(self):
        if self.model is None:
            self.show_message('请先选择模型yaml配置文件.')
            
        if self._timer is not None:
            self.reset_timer()
        
        if type(self.now) is cv2.VideoCapture:
        
            """
            处理视频流
            """
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(self.now.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.now.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if self.out is None:
                self.out = cv2.VideoWriter(os.path.join(self.save_path, f'{self.save_id}.mp4'), fourcc, 30.0, size)
            
            self.track_init()
            # self._timer = QTimer(self)
            # 启动新的线程来处理视频帧
            video_thread = threading.Thread(target=self.show_video)
            video_thread.start()
            # self._timer.timeout.connect(self.show_video)
            # self._timer.start(20)
        elif type(self.now) is list:
            """
            处理列表图像
            """
            self.print_id, self.folder_len = 1, len(self.now)
            self._timer = QTimer(self)
            self._timer.timeout.connect(self.show_folder)
            self._timer.start(20)
        else:
            """
            处理单张图像
            """
            torch.cuda.synchronize()
            since = time.time()
            result, image_det = detect_image(model=self.model, image=self.now, name_classes=self.cfg['name_classes'], num_classes=self.cfg['num_classes'], input_shape=self.cfg['input_shape'], device=self.device, weight_type=self.weight_type)
            torch.cuda.synchronize()
            end = time.time()
            cv2.imencode(".jpg", image_det)[1].tofile(os.path.join(self.save_path, f'{self.save_id}.jpg'))
            self.ui.textBrowser.append(f'time:{end-since:.5f}s save image in {os.path.join(self.save_path, f"{self.save_id}.jpg")}\n' + self.info)
            self.save_id += 1
            self.show_image_from_array(image_det, det=True)
    
    def stop_detect(self):
        if self._timer is not None:
            self.reset_timer()
    
    def select_savepath(self):
        folder = QFileDialog.getExistingDirectory(self, '选择路径', '.')
        self.save_path = folder
        self.save_id = 0
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
    
    def show_folder(self):
        if len(self.now) == 0:
            self.reset_timer()
        else:
            img_path = self.now[0]
            image = self.read_and_show_image_from_path(img_path)
            torch.cuda.synchronize()
            since = time.time()
            result, image_det = detect_image(model=self.model, image=image, name_classes=self.cfg['name_classes'], num_classes=self.cfg['num_classes'], input_shape=self.cfg['input_shape'], device=self.device, weight_type=self.weight_type)
            torch.cuda.synchronize()
            end = time.time()
            cv2.imencode(".jpg", image_det)[1].tofile(os.path.join(self.save_path, f'{self.save_id}.jpg'))
            # self.ui.textBrowser.append(f'time:{end-since:.5f}s {self.print_id}/{self.folder_len} save image in {os.path.join(self.save_path, f"{self.save_id}.jpg")}')
            self.text_update_signal.emit(f'time:{end-since:.5f}s {self.print_id}/{self.folder_len} save image in {os.path.join(self.save_path, f"{self.save_id}.jpg")}')
            self.show_image_from_array(image_det, det=True)
            self.print_id += 1
            self.save_id += 1
            self.now.pop(0)
    
    
    def show_video(self):
        while self.now is not None:
            flag, image = self.now.read()
            if flag:
                self.show_image_from_array(image, ori=True)
                torch.cuda.synchronize()
                since = time.time()
                seg_img, image_det = detect_image(model=self.model, image=image, name_classes=self.cfg['name_classes'], num_classes=self.cfg['num_classes'], input_shape=self.cfg['input_shape'], device=self.device, weight_type=self.weight_type)
                if self.ui.comboBox.currentText() != 'NoTrack':
                    image_det = self.model.track_processing(image.copy(), seg_img)
                torch.cuda.synchronize()
                end = time.time()
                self.out.write(image_det)
                self.show_image_from_array(image_det, det=True)
                if self.video_count is not None:
                    self.text_update_signal.emit(f'{self.print_id}/{self.video_count} Frames. time:{end-since:.5f}s fps:{1 / (end-since):.3f}' + self.info)
                    # self.ui.textBrowser.append(f'{self.print_id}/{self.video_count} Frames. time:{end-since:.5f}s fps:{1 / (end-since):.3f}' + self.info)
                else:
                    self.text_update_signal.emit(f'{self.print_id} Frames. time:{end-since:.5f}s fps:{1 / (end-since):.3f}' + self.info)
                    # self.ui.textBrowser.append(f'{self.print_id} Frames. time:{end-since:.5f}s fps:{1 / (end-since):.3f}' + self.info)
                self.print_id += 1
                
            else:
                self.now = None
                self.reset_timer()
                self.out.release()
                self.out = None
                self.reset_video_count()
                self.save_id += 1

    def _exit(self):
        self.close()

if __name__ == '__main__':
    gui_title = 'FastSegFormer-VisionSystem'
    textBrowser_size = 15
    
    app = QApplication(sys.argv)
    w = MyForm(title=gui_title, textBrowser_size=textBrowser_size)
    sys.exit(app.exec_())