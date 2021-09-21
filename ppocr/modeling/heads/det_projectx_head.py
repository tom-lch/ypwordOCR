# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

def get_bias_attr(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr


class Head(nn.Layer):
    def __init__(self, in_channels, name_list):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')
        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=2,
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=2,
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x

class ThreshHead(nn.Layer):
    def __init__(self, in_channels, name_list):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')
        self.MaxPooling1 = nn.MaxPool2D(
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=2,
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=2,
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )
        # 添加双线插值
        self.ROIAlign = nn.UpsamplingBilinear2D(scale_factor=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.MaxPooling1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.conv3(x)
        x = self.ROIAlign(x)
        x = F.relu(x)
        return x

class ProjectXHead(nn.Layer):
    """
    args:
        params(dict): super parameters for build ProjectX network
    """

    def __init__(self, in_channels, k=10, **kwargs):
        super(ProjectXHead, self).__init__()
        self.k = k
        binarize_name_list = [
            'conv2d_56', 'batch_norm_47', 'conv2d_transpose_0', 'batch_norm_48',
            'conv2d_transpose_1', 'binarize'
        ]
        thresh_name_list = [
            'conv2d_57', 'batch_norm_49', 'max_pool_111','conv2d_transpose_2', 'batch_norm_50',
            'conv2d_transpose_3', 'upsampling_bilinear_2d_1','thresh'
        ]
        
        self.binarize = Head(in_channels, binarize_name_list)
        # 不在使用自适应阈值图
        self.thresh = ThreshHead(in_channels, thresh_name_list)

    def step_function(self, x, y):
        # 将特征图进行划分，如果特征较为明显，使用这块区域中的均值>0.5 使用1-均值，否则使用均值填充。
        # 为了便于求阈值，将图像划分成若干个子图
        paddle.to_tensor()
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x-y)))

    def forward(self, x, targets=None):
        shrink_maps = self.binarize(x)
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([threshold_maps, binary_maps], axis=1)
        return {'maps': y}
