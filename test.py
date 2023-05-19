from models.fastsegfomer.fastsegformer import FastSegFormer
from utils.utils import *
import torch
import cv2
import onnxruntime

def pred_img():
    image = cv2.imdecode(np.fromfile('images/img236.jpg', np.uint8), cv2.IMREAD_COLOR)
    model_path = 'weights/FastSegFormer_P_224.pth'
    net = FastSegFormer(num_classes=4, pretrained=False, backbone='poolformer_s12',Pyramid="multiscale", cnn_branch=True).to(device='cuda')
    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    result, image_det = detect_image(model=net, image = image, name_classes=["background", "sunburn", "Ulcer", "wind scarring"], num_classes=4, input_shape=[224, 224], device='cuda', weight_type='pth')
    
    cv2.imshow("x", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("x", image_det)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def pred_img_onnx():
    image = cv2.imdecode(np.fromfile('images/img236.jpg', np.uint8), cv2.IMREAD_COLOR)
    model_path = 'weights/FastSegFormer_P_224.onnx'
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    net = onnxruntime.InferenceSession(model_path, providers=providers)
    result, image_det = detect_image(model=net, image = image, name_classes=["background", "sunburn", "Ulcer", "wind scarring"], num_classes=4, input_shape=[224, 224], device='cuda', weight_type='onnx')
    
    cv2.imshow("x", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("x", image_det)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def simplify_onnx_model():
    import onnxsim
    import onnx
    print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
    
    model_path = "weights/FastSegFormer_P_224.onnx"
    model_onnx = onnx.load(model_path)
    
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        dynamic_input_shape=False,
        input_shapes=None)
    assert check, 'assert check failed'
    onnx.save(model_onnx, "weights/FastSegFormer_P_224_simplify.onnx")
    

def fp16_pth2onnx_model():
    import onnx
    
    model_path = "weights/FastSegFormer_P_224_FP16.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FastSegFormer(num_classes=4, pretrained=False, Pyramid='multiscale', cnn_branch=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    im = torch.randn(1, 3, 224, 224)
    input_layer_names = ["images"]
    output_layer_names = ["output"]
    
    model_out_path = "weights/FastSegFormer_P_224_FP16.onnx"
        
    # Export the model
    print(f'Starting export with onnx {onnx.__version__}.')
    torch.onnx.export(model,
                    im,
                    f               = model_out_path,
                    verbose         = False,
                    opset_version   = 12,
                    training        = torch.onnx.TrainingMode.EVAL,
                    do_constant_folding = True,
                    input_names     = input_layer_names,
                    output_names    = output_layer_names,
                    dynamic_axes    = None)

    # Checks
    model_onnx = onnx.load(model_out_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model


def make_orange_mp4():
    import cv2
    import numpy as np
    import os

    # 设置图像尺寸和行列数
    image_width = 512
    image_height = 512
    rows = 1
    cols = 20
    time = 60
    
    # 读取图像
    images = []
    images_folder = "images"
    image_file_list = os.listdir(images_folder)
    for i in range(len(image_file_list)):
        image_path = os.path.join(images_folder, image_file_list[i])
        image = cv2.imread(image_path)
        # image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
        images.append(image)

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "video/output_video.mp4"  # 请替换为你的输出路径
    fps = 30
    out = cv2.VideoWriter(output_path, fourcc, fps, (image_width, image_height))
    step = (image_width * (cols - 1) * rows) // (time * fps)

    # 生成视频帧
    frame_list = []
    for i in range(cols - 1):
        # print(i)
        frame = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        # 每次合并两张图像
        img_merge = np.concatenate([images[i], images[i + 1]], axis=1)
        for j in range(image_width//step):
            # print(j)
            frame = img_merge[:, j*step : image_width+j*step, :]
            # 写入视频帧
            frame_list.append(frame)
            
    # 写入视频帧
    for frame in frame_list:
        out.write(frame)

    # 释放资源
    out.release()

    
    





    
    
    
if __name__ == '__main__':
    pred_img()
    # pred_img_onnx()
    # simplify_onnx_model()
    # fp16_pth2onnx_model()
    # make_orange_mp4()


