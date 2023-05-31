import cv2
import copy
import numpy as np
import time
import threading
from tqdm import tqdm
import os
import argparse
import yaml

import onnxruntime

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit 

import torch
import torch.nn.functional as F

from models.fastsegformer.fastsegformer import FastSegFormer




#设置gstreamer管道参数
def gstreamer_pipeline(
    capture_width=1280, #摄像头预捕获的图像宽度
    capture_height=720, #摄像头预捕获的图像高度
    display_width=1280, #窗口显示的图像宽度
    display_height=720, #窗口显示的图像高度
    framerate=60,       #捕获帧率
    flip_method=0,      #是否旋转图像
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(TRT_LOGGER, max_batch_size=1, onnx_file_path="", engine_file_path="",fp16_mode=True, save_engine=False):
    """
    params max_batch_size:      预先指定大小好分配显存
    params onnx_file_path:      onnx文件路径
    params engine_file_path:    待保存的序列化的引擎文件路径
    params fp16_mode:           是否采用FP16
    params save_engine:         是否保存引擎
    returns:                    ICudaEngine
    """
    # 如果已经存在序列化之后的引擎，则直接反序列化得到cudaEngine
    if os.path.exists(engine_file_path):
        print("Reading engine from file: {}".format(engine_file_path))
        with open(engine_file_path, 'rb') as f, \
            trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化
    else:  # 由onnx创建cudaEngine
        
        # 使用logger创建一个builder 
        # builder创建一个计算图 INetworkDefinition
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.

        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network,  \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            builder.create_builder_config() as config, \
            trt.Runtime(TRT_LOGGER) as runtime: # 使用onnx的解析器绑定计算图，后续将通过解析填充计算图
            
            config.max_workspace_size = 1<<30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
            builder.max_batch_size = max_batch_size # 执行时最大可以使用的batchsize
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)

            # 解析onnx文件，填充计算图
            if not os.path.exists(onnx_file_path):
                quit("ONNX file {} not found!".format(onnx_file_path))
            print('loading onnx file from path {} ...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model: # 二值化的网络结果和参数
                print("Begining onnx file parsing")
                parser.parse(model.read())  # 解析onnx文件
            #parser.parse_from_file(onnx_file_path) # parser还有一个从文件解析onnx的方法

            print("Completed parsing of onnx file")
            # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
            print("Building an engine from file {}' this may take a while...".format(onnx_file_path))

            #################
            # print(network.get_layer(network.num_layers-1).get_output(0).shape)
            # network.mark_output(network.get_layer(network.num_layers -1).get_output(0))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan) 
            print("Completed creating Engine")
            if save_engine:  #保存engine供以后直接反序列化使用
                with open(engine_file_path, 'wb') as f:
                    f.write(engine.serialize())  # 序列化
            return engine


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def mkdir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)
        print("成功创建文件夹: {directory}" )



def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


def preprocess_input(image):
    image /= 255.0
    return image

def blend_images(old_image, new_image, alpha):
    """
    使用cv2.addWeighted()函数混合两个图像
    """
    blended_image = cv2.addWeighted(old_image, alpha, new_image, 1 - alpha, 0)

    return blended_image


def process_frame(model, img, name_classes = None, num_classes = 21, count = False, input_shape = (224, 224), device = 'cpu', weight_type = None, inputs=None, outputs=None, bindings=None, stream=None, context=None):
    
    # torch.cuda.synchronize()
    since = time.time()
    
    # 转化为彩色图像
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 对输入图像做一个备份
    old_img = copy.deepcopy(img)
    original_h  = np.array(img).shape[0]
    original_w  = np.array(img).shape[1]
    # 将图像转化为模型输入的分辨率
    if original_h != input_shape[0] or original_w != input_shape[1]: 
        image_data = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
    # 添加Batch维度
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    # 将内存不连续的数组转成连续存储
    image_data = np.ascontiguousarray(image_data)

    if weight_type == '.onnx':
        ort_inputs = {'images': image_data}
        pred = model.run(['output'], ort_inputs)[0]
        pred = pred[0]
        # 转化为张量
        pred = torch.tensor(pred)
        pred = F.softmax(pred.permute(1,2,0),dim = -1).cpu().numpy()
        pred = cv2.resize(pred, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
        pred = pred.argmax(axis=-1)
    
    elif weight_type == '.trt':
        # 输出为一个列表
        shape_of_output = (1, 4, 224, 224)
        # Load data to the buffer
        inputs[0].host = image_data.reshape(-1)
        trt_outputs = do_inference_v2(context=context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = postprocess_the_outputs(trt_outputs[0], shape_of_output)
        pred = pred[0]
        # 转化为张量
        pred = torch.tensor(pred)
        pred = F.softmax(pred.permute(1,2,0),dim = -1).cpu().numpy()
        pred = cv2.resize(pred, (original_w, original_h), interpolation = cv2.INTER_LINEAR)
        pred = pred.argmax(axis=-1)

    if count:
            classes_nums        = np.zeros([num_classes])
            total_points_num    = original_h * original_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(num_classes):
                num     = np.sum(pred == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        
    if num_classes <= 21:
        colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                        (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                        (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                        (128, 64, 12)]

    # 转回原图尺寸    
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pred, [-1])], [original_h, original_w, -1])
    # 分割图像和原图结合
    image = blend_images(old_image=old_img, new_image=seg_img, alpha=0.6)
    # 转回RGB图像    
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # torch.cuda.synchronize()
    end = time.time()

    # 计算FPS并且放到图像的左上角
    fps = 1 / (end - since)
    image = cv2.putText(image, "FPS " + str(int(fps)), (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, cv2.LINE_AA)
    fps = int(fps)
            
    return seg_img, image, fps


def cv2_camera_thread(model=None, name_classes=None, num_classes = 21, count = False, input_shape = [224, 224], device = 'cpu', weight_type = None, thread = False,
                      inputs=None, outputs=None, bindings=None, stream=None, context=None):
    
    # 设置CSI摄像头参数
    capture_width = 1280
    capture_height = 720
    display_width = 640
    display_height = 480
    framerate = 30
    flip_method = 0

    # 创建管道
    print(gstreamer_pipeline(capture_width,capture_height,display_width,display_height,framerate,flip_method))
    # 管道与视频流绑定
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) # CIS摄像头

    if cap.isOpened():
        window_handle = cv2.namedWindow("FastSegFormer-Detection-System", cv2.WINDOW_AUTOSIZE)
        print("摄像头正常打开！")

        frame_num = 0
        fps_all = 0
        while cv2.getWindowProperty("FastSegFormer-Detection-System", 0) >= 0:
            flag, frame = cap.read()
            if not flag:
                break
        
            frame_num += 1
            # 处理帧函数交给新的线程
            if thread:

                def process_frame_in_thread(model, frame, name_classes, num_classes, count, input_shape, device, weight_type, fps_all):
                    _, frame, fps = process_frame(model, frame, name_classes, num_classes, count, input_shape, device, weight_type,
                                                  inputs=inputs, outputs=outputs, bindings=bindings, stream=stream, context=context)
                    fps_all += fps


                # 创建新线程并运行处理帧函数
                thread = threading.Thread(target=process_frame_in_thread, args=(model, frame, name_classes, num_classes, count, input_shape, device, weight_type, fps_all))
                thread.start()
            else:
                _, frame, fps = process_frame(model, frame, name_classes, num_classes, count, input_shape, device, weight_type,
                                              inputs=inputs, outputs=outputs, bindings=bindings, stream=stream, context=context)
                fps_all += fps
            cv2.imshow('FastSegFormer-Detection-System', frame)
            # 按键盘上的q或者esc退出
            if cv2.waitKey(1) in [ord('q'), 27]:
                break
        # 关闭摄像头
        cap.release()
        # 关闭图像窗口
        cv2.destroyAllWindows()
        print("摄像头检测的平均帧数为: " + str(int(fps_all / frame_num - 1)))
    else:
        print("打开摄像头失败！")


def cv2_video_thread(model=None, video_path=None, name_classes = None, num_classes = 21, count = False, input_shape = [224, 224], device = 'cpu', weight_type = None, thread = False, view = False,
                     inputs=None, outputs=None, bindings=None, stream=None, context=None):

    if weight_type == '.onnx':
        outhead = 'orange-onnx-fp16.mp4'
    elif weight_type == '.trt':
        outhead = 'orange-trt-fp16.mp4'
    out_folder = "results"
    mkdir(out_folder)
    output_path = os.path.join("results", outhead)
    
    print('视频开始处理', video_path)
    
    # 获取视频总帧数
    cap = cv2.VideoCapture(video_path) # 获取视频 

    assert cap.isOpened(), f'Failed to open {video_path}'

    frame_count = 0
    while(cap.isOpened()):
        flag, frame = cap.read()
        frame_count += 1
        if not flag:
            break
    cap.release()
    print('视频总帧数为',frame_count)
    
    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Failed to open {video_path}'
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("视频的分辨率为", frame_size)

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    fps_all = 0

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count-1) as pbar:
        try:
            while(cap.isOpened()):
                flag, frame = cap.read()
                if not flag:
                    break

                # 处理帧
                try:
                    if thread:

                        def process_frame_in_thread(model, frame, name_classes, num_classes, count, input_shape, device, weight_type, view, fps_all):
                            _, frame, fps = process_frame(model, frame, name_classes, num_classes, count, input_shape, device, weight_type,
                                                          inputs=inputs, outputs=outputs, bindings=bindings, stream=stream, context=context)
                            fps_all += fps
                        
                        # 创建新线程并运行处理帧函数以及观看、保存视频
                        thread = threading.Thread(target=process_frame_in_thread, args=(model, frame, name_classes, num_classes, count, input_shape, device, weight_type, view, fps_all))
                        thread.start()
                    else:
                        _, frame, fps = process_frame(model, frame, name_classes, num_classes, count, input_shape, device, weight_type,
                                                        inputs=inputs, outputs=outputs, bindings=bindings, stream=stream, context=context)
                        fps_all += fps
                except:
                    print('报错！', f'Error')
                    pass
                
                if flag == True:
                    if view:
                        cv2.imshow('FastSegFormer-Detection-System', frame)
                    out.write(frame)
                    # 进度条更新一帧
                    pbar.update(1)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('检测视频的平均帧数为: ' + str(int(fps_all / frame_count - 1)))
    print('视频已保存', output_path)




def parse_args():
    parser = argparse.ArgumentParser(description='FastSegFormer inference test')
    
    parser.add_argument('--cfg',
                        help='model configure file name',
                        default="configs/fastsegformer_p_224_FP16_onnx.yaml",
                        type=str)
    parser.add_argument('--videoPath',
                        help='video file path',
                        default="video/orange.mp4",
                        type=str)
    parser.add_argument('--mode',
                        help='camera/video mode',
                        default='camera',
                        type=str) 
    parser.add_argument('--weight_type',
                        help='weight file format',
                        default='.onnx',
                        type=str)
    parser.add_argument('--thread',
                        help='multi-threading',
                        default=False,
                        type=bool)
    parser.add_argument('--view',
                        help='Visualization when mode is video',
                        default=False,
                        type=bool)

    return parser.parse_args()




def main(args):

    # read cfg
    with open(args.cfg) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # 一些参数配置
    num_classes = config['num_classes']
    model_path = config['model_path']
    model_type = config['model_type']
    input_shape = (224, 224)
    name_classes = config['name_classes']
    video_path = args.videoPath
    mode = args.mode
    weight_type = args.weight_type
    thread = args.thread
    view = args.view

    # 推理设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化推理引擎
    if weight_type == '.onnx':
        # 在jetson系列产品上不能只能通过pip安装onnxruntime-gpu，需要手动编译，也可以直接下载NVIDIA官方提供的编译版本
        # https://elinux.org/Jetson_Zoo#ONNX_Runtime 提供了不同jetpack和python版本的arm64编译包 
        # 环境信息： CUDA：10.2 jetpack：4.6 python：3.6 ==> onnxruntime-gpu: 1.6.0
        if model_type == 'FastSegFormer_P':
            model = FastSegFormer(num_classes, pretrained=False, backbone='poolformer_s12', Pyramid='multiscale', cnn_branch=True).to(device).eval()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            model = onnxruntime.InferenceSession(model_path, providers=providers)
        
        # 开始任务
        if mode == 'camera':
            cv2_camera_thread(model, name_classes, num_classes, count=False, input_shape=input_shape, device=device, weight_type=weight_type, thread=thread)
        elif mode == 'video':
            cv2_video_thread(model, video_path, name_classes, num_classes, count = False, input_shape=input_shape, device=device, weight_type=weight_type, thread=thread, view=view)
    elif weight_type == '.trt':
        TRT_LOGGER = trt.Logger()
        eng_path = model_path.split(".")[0] + ".trt"
        # 构建engine
        engine = get_engine(TRT_LOGGER, onnx_file_path=model_path, engine_file_path=eng_path)
        # 构建Context
        context = engine.create_execution_context()
        # 从engine中获取size并分配buffer（尤其是动态模式）
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        # 开始任务
        if mode == 'camera':
            cv2_camera_thread(model=None,name_classes=name_classes, num_classes=num_classes, count=False, input_shape=input_shape, device=device, weight_type=weight_type, thread=thread,
                                inputs=inputs, outputs=outputs, bindings=bindings, stream=stream, context=context)
        elif mode == 'video':
            cv2_video_thread(model=None,video_path=video_path, name_classes=name_classes, num_classes=num_classes, count = False, input_shape=input_shape, device=device, weight_type=weight_type, thread=thread, view=view,
                                inputs=inputs, outputs=outputs, bindings=bindings, stream=stream, context=context)
        
    
    

def test():
    video_path = 'video/orange.mp4'

    cap = cv2.VideoCapture(video_path)
    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        # 检查是否成功读取帧
        if not ret:
            break
        # 在窗口中显示帧
        cv2.imshow("Video", frame)
        # 按下键盘上的q键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频文件资源
    cap.release()
    # 关闭窗口
    cv2.destroyAllWindows()



if __name__ == '__main__':
    args = parse_args()
    main(args)
    # test()
