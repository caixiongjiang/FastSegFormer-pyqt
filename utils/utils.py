import numpy as np
from PIL import Image
import copy
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F


def cvtColor(image):
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image 
        else:
            image = image.convert('RGB')
            return image 

def preprocess_input(image):
    image /= 255.0
    return image



def blend_images(old_image, new_image, alpha):
    """
    使用cv2.addWeighted()函数混合两个图像
    """
    blended_image = cv2.addWeighted(old_image, alpha, new_image, 1 - alpha, 0)

    return blended_image



def detect_image(model, image, name_classes = None, num_classes = 21, count = False, mix_type = 0, input_shape = [224, 224], device = 'cpu'):
        # 转化为彩色图像
        image = cvtColor(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 对输入图像做一个备份
        old_img = copy.deepcopy(image)
        original_h  = np.array(image).shape[0]
        original_w  = np.array(image).shape[1]
        
        image_data = cv2.resize(image, input_shape, interpolation=cv2.INTER_LINEAR)
        # 添加Batch维度
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            
            images = images.to(device)
            pred = model(images)[0]
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

        if mix_type == 0:
            seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pred, [-1])], [original_h, original_w, -1])
            image = blend_images(old_image=old_img, new_image=seg_img, alpha=0.6)
        elif mix_type == 1:
            seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pred, [-1])], [original_h, original_w, -1])
            image = seg_img
            
        return old_img, image