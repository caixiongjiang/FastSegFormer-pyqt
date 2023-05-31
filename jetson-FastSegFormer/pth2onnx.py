#--*-- coding:utf-8 --*--
import torch
from models.fastsegformer.fastsegformer import FastSegFormer

net = FastSegFormer(num_classes=4, pretrained=False, backbone='poolformer_s12', Pyramid='multiscale', cnn_branch=True).cuda()
net.eval()

export_onnx_file = "weights/FastSegFormer_P_224_FP16.onnx"
x=torch.onnx.export(net,  # 待转换的网络模型和参数
                torch.randn(1, 3, 224, 224, device='cuda'), # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                export_onnx_file,  # 输出文件的名称
                verbose=False,      # 是否以字符串的形式显示计算图
                input_names=["images"],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                output_names=["output"], # 输出节点的名称
                opset_version=11,   # onnx 支持采用的operator set, 应该和pytorch版本相关，目前我这里最高支持10
                do_constant_folding=True, # 是否压缩常量
                dynamic_axes=None)

