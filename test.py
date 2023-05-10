from models.fastsegfomer.fastsegformer import FastSegFormer
from utils.utils import *
import torch
import cv2
from run_gui import resize_img

image = cv2.imdecode(np.fromfile('images\img236.jpg', np.uint8), cv2.IMREAD_COLOR)


model_path = 'weights/FastSegFormer_P_224.pth'
net = FastSegFormer(num_classes=4, pretrained=False, backbone='poolformer_s12',Pyramid="multiscale", cnn_branch=True).to(device='cuda')
net.load_state_dict(torch.load(model_path, map_location='cuda'))

old_img, result = detect_image(model=net, image = image, name_classes=["background", "sunburn", "Ulcer", "wind scarring"], num_classes=4, input_shape=[224, 224], device='cuda')
old_img = cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR)
result = cv2.cvtColor(resize_img(result), cv2.COLOR_RGB2BGR)


cv2.imshow("x", old_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("x", result)
cv2.waitKey(0)
cv2.destroyAllWindows()



