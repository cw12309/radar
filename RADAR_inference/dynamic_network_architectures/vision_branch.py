# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import sys
from monai.utils import deprecated_arg
import pydoc
import warnings
from typing import Union
import torch.nn.functional as F


class VisionBranch(nn.Module):
    def __init__(
        self,
        in_channels=1
    ) -> None:
        super().__init__()

        # define LightDecoderUNet
        self.UNet = self.get_network_from_plans(
            arch_class_name="dynamic_network_architectures.architectures.unet_lightdecoder.PlainConvUNetLightD",
            arch_kwargs={
                "n_stages": 6,
                "features_per_stage": [32, 64, 128, 256, 320, 320],
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "kernel_sizes": [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "strides": [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
                "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
                "conv_bias": True,
                "norm_op": "torch.nn.BatchNorm3d",
                "norm_op_kwargs": {},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.ReLU",
                "nonlin_kwargs": {"inplace": True},
            },
            arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
            input_channels=1,
            output_channels=37,
            allow_init=True,
            deep_supervision=True,
        )
        
        self.proj1 = nn.Conv3d(320, 256, kernel_size=1)
        self.proj2 = nn.Conv3d(320, 256, kernel_size=1)
        self.proj3 = nn.Conv3d(256, 256, kernel_size=1)
        
        self.organs = [
            '肾上腺', '主动脉', '竖脊肌', '脑', '锁骨', '大肠', '十二指肠', '食管', '面部', '股骨', 
            '胆囊', "臀肌", '心脏', '髋关节', '肱骨', '髂动脉', '髂静脉', '髂腰肌', '下腔静脉', '肾', 
            '肝', '肺', '胰腺', '门静脉', '肺动脉', '肋骨', '骶骨', '肩胛骨', '小肠', '脾', 
            '胃', '气管', '膀胱', '颈椎', '腰椎', '胸椎'
        ]
        # organ_dict = {
        #     "肾上腺": "adrenal gland", "主动脉": "aorta", "竖脊肌": "erector spinae muscle", "脑": "brain", "锁骨": "clavicle", "大肠": "large bowel", "十二指肠": "duodenum", 
        #     "食管": "esophagus", "面部": "face", "股骨": "femur", "胆囊": "gallbladder", "臀肌": "gluteus muscle", "心脏": "heart", "髋关节": "hip joint", "肱骨": "humerus", 
        #     "髂动脉": "iliac artery", "髂静脉": "iliac vena", "髂腰肌": "iliopsoas muscle", "下腔静脉": "inferior vena cava", "肾": "kidney", "肝": "liver", "肺": "lung", 
        #     "胰腺": "pancreas", "门静脉": "portal vein", "肺动脉": "pulmonary artery", "肋骨": "rib", "骶骨": "sacrum", "肩胛骨": "scapula", "小肠": "small bowel", "脾": "spleen", 
        #     "胃": "stomach", "气管": "trachea", "膀胱": "bladder", "颈椎": "cervical vertebrae", "腰椎": "lumbar vertebrae", "胸椎": "thoracic vertebrae"
        # }

    def get_network_from_plans(sefl, arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
        network_class = arch_class_name
        architecture_kwargs = dict(**arch_kwargs)
        for ri in arch_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

        nw_class = pydoc.locate(network_class)

        if deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = deep_supervision

        network = nw_class(
            input_channels=input_channels,
            num_classes=output_channels,
            **architecture_kwargs
        )

        if hasattr(network, 'initialize') and allow_init:
            network.apply(network.initialize)

        return network

    def forward(self, x, y):
        # fp16 autocast 包住视觉主干（UNet + 投影 + 上采样），这是单窗前向显存
        # 峰值的大头：fp32 实测 reserved 6.04 GB，推理时冲到 7.5~7.9 GB，贴顶
        # 8 GB 显存上限，余量仅数百 MB，导致 driver 级 "CUDA driver error: out
        # of memory"（表现为非确定性失败：同一固定 roi_size 窗口，有时过有时不过）。
        # fp16 实测 reserved 3.80 GB，彻底脱离临界区。
        # 对外返回值统一 .float() 转回 fp32，使下游 attention、视觉-文本匹配
        # （logits/temp 放大后对精度敏感）与 full_mask 累加全部仍在 fp32 下计算，
        # 置信度数值不受半精度影响。
        # CPU 路径 enabled=False，完全走原 fp32 分支（CPU 上 fp16 卷积慢且不被支持）。
        with torch.autocast('cuda', dtype=torch.float16, enabled=x.is_cuda):
            skips, segs = self.UNet(x)

            scale1 = self.proj1(skips[-1])
            scale2 = self.proj2(skips[-2])
            scale3 = self.proj3(skips[-3])

            # process mask
            pred_logit = segs[0]
            # autocast 把 softmax 列为 fp32 算子，seg_probs 本就是 fp32；
            # 仍显式 .float() 兜底，保证 full_mask 累加侧 dtype 恒定。
            seg_probs = torch.softmax(pred_logit, 1).float()
            target_size = [pred_logit.shape[-3], pred_logit.shape[-2]*2, pred_logit.shape[-1]*2]
            pred_logit = F.interpolate(pred_logit, size=target_size, mode='trilinear', align_corners=False)
            # 直接对 logits 取 argmax：softmax 沿通道维单调，不改变 argmax 结果，
            # 省掉一份全窗口 (1,37,96,256,384) float32 张量（实测 1.30 GB 显存）。
            # 该 mask 仅在下方 no_grad 块里做 max_pool3d 生成 organ_token_flags，
            # 不需要概率值；对外返回的 seg_probs 仍由上面低分辨率 softmax 提供。
            pred_mask = pred_logit.argmax(1)
            y = pred_mask

            # embeds 是 Linear(autocast→fp16) 输出，转回 fp32 供下游 attention 使用，
            # 否则 fp16 输入配 fp32 权重的 MultiheadAttention 会 dtype 不匹配。
            res_x1 = scale1.flatten(2).transpose(1, 2).float()
            res_x2 = scale2.flatten(2).transpose(1, 2).float()
            res_x3 = scale3.flatten(2).transpose(1, 2).float()
        
        B, L1, _ = res_x1.size()
        B, L2, _ = res_x2.size()
        B, L3, _ = res_x3.size()
        
        with torch.no_grad():
            organ_token_flags1 = torch.zeros(B, len(self.organs), L1, dtype=bool).to(x.device)
            organ_token_flags2 = torch.zeros(B, len(self.organs), L2, dtype=bool).to(x.device)
            organ_token_flags3 = torch.zeros(B, len(self.organs), L3, dtype=bool).to(x.device)
            
            b = x.size(0)
            for i in range(b):
                unique_values = torch.unique(y[i])
                unique_values = unique_values[unique_values != 0]
                if unique_values.tolist() == []:
                    continue
                # 逐器官处理：max_pool3d 把 N 个器官掩码当 batch 维独立池化，
                # 拆成单器官循环结果完全等价，但把 (N,96,256,384) float32 的
                # 峰值占用从 N×37.7MB 降到 1×37.7MB。大 FOV 影像（如 KiTS
                # 495×495）单窗检出器官多，原先批量 stack 会顶爆显存。
                for uv in unique_values:
                    organ_idx = int(uv) - 1
                    m = torch.eq(y[i], uv).float()[None, None]  # (1,1,D,H,W)
                    h3 = F.max_pool3d(
                        m, kernel_size=(2, 8, 8), stride=(2, 8, 8)
                    ).flatten(1)[0] > 0
                    h2 = F.max_pool3d(
                        m, kernel_size=(4, 16, 16), stride=(4, 16, 16)
                    ).flatten(1)[0] > 0
                    h1 = F.max_pool3d(
                        m, kernel_size=(8, 32, 32), stride=(8, 32, 32)
                    ).flatten(1)[0] > 0
                    organ_token_flags1[i][organ_idx] = h1
                    organ_token_flags2[i][organ_idx] = h2
                    organ_token_flags3[i][organ_idx] = h3

        return seg_probs, pred_mask, res_x1, res_x2, res_x3, organ_token_flags1, organ_token_flags2, organ_token_flags3
