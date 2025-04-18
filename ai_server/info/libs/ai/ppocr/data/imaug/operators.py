"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import six
import cv2
import numpy as np
import math
from PIL import Image
from mylogger import logger


class DecodeImage(object):
    """ decode image """

    def __init__(self,
                 img_mode='RGB',
                 channel_first=False,
                 ignore_orientation=False,
                 **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data, **kwargs):
        img = data['image']
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION |
                               cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data, **kwargs):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
                                img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data, **kwargs):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


# class Fasttext(object):
#     def __init__(self, path="None", **kwargs):
#         import fasttext
#         self.fast_model = fasttext.load_model(path)
#
#     def __call__(self, data):
#         label = data['label']
#         fast_label = self.fast_model[label]
#         data['fast_label'] = fast_label
#         return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data, **kwargs):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class Pad(object):
    def __init__(self, size=None, size_div=32, **kwargs):
        if size is not None and not isinstance(size, (int, list, tuple)):
            raise TypeError("Type of target_size is invalid. Now is {}".format(
                type(size)))
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.size_div = size_div

    def __call__(self, data, **kwargs):

        img = data['image']
        img_h, img_w = img.shape[0], img.shape[1]
        if self.size:
            resize_h2, resize_w2 = self.size
            assert (
                    img_h < resize_h2 and img_w < resize_w2
            ), '(h, w) of target size should be greater than (img_h, img_w)'
        else:
            resize_h2 = max(
                int(math.ceil(img.shape[0] / self.size_div) * self.size_div),
                self.size_div)
            resize_w2 = max(
                int(math.ceil(img.shape[1] / self.size_div) * self.size_div),
                self.size_div)
        img = cv2.copyMakeBorder(
            img,
            0,
            resize_h2 - img_h,
            0,
            resize_w2 - img_w,
            cv2.BORDER_CONSTANT,
            value=0)
        data['image'] = img
        return data


class Resize(object):
    def __init__(self, size=(640, 640), **kwargs):
        self.size = size

    def resize_image(self, img):
        resize_h, resize_w = self.size
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]

    def __call__(self, data, **kwargs):
        img = data['image']
        if 'polys' in data:
            text_polys = data['polys']

        img_resize, [ratio_h, ratio_w] = self.resize_image(img)
        if 'polys' in data:
            new_boxes = []
            for box in text_polys:
                new_box = []
                for cord in box:
                    new_box.append([cord[0] * ratio_w, cord[1] * ratio_h])
                new_boxes.append(new_box)
            data['polys'] = np.array(new_boxes, dtype=np.float32)
        data['image'] = img_resize
        return data


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.keep_ratio = False
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
            if 'keep_ratio' in kwargs:
                self.keep_ratio = kwargs['keep_ratio']
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data, **kwargs):
        img = data['image']
        src_h, src_w, _ = img.shape
        if sum([src_h, src_w]) < 64:
            img = self.image_padding(img)

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img, **kwargs)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img, **kwargs)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img, **kwargs)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def image_padding(self, im, value=0):
        h, w, c = im.shape
        im_pad = np.zeros((max(32, h), max(32, w), c), np.uint8) + value
        im_pad[:h, :w, :] = im
        return im_pad

    def resize_image_type1(self, img, **kwargs):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        if self.keep_ratio is True:
            resize_w = ori_w * resize_h / ori_h
            N = math.ceil(resize_w / 32)
            resize_w = N * 32
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img, det_limit_side_len=None, **kwargs):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        if not isinstance(det_limit_side_len, int):
            det_limit_side_len = self.limit_side_len

        logger.info({'det_limit_side_len': det_limit_side_len})

        h, w, c = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > det_limit_side_len:
                if h > w:
                    ratio = float(det_limit_side_len) / h
                else:
                    ratio = float(det_limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'min':
            if min(h, w) < det_limit_side_len:
                if h < w:
                    ratio = float(det_limit_side_len) / h
                else:
                    ratio = float(det_limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'resize_long':
            ratio = float(det_limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type, image ')
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img, **kwargs):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, **kwargs):
        img = img.resize(self.size, self.interpolation)
        img_numpy = np.array(img).astype("float32")
        img_numpy = img_numpy.transpose((2, 0, 1)) / 255
        return img_numpy


class GrayImageChannelFormat(object):
    """
    format gray scale image's channel: (3,h,w) -> (1,h,w)
    Args:
        inverse: inverse gray image
    """

    def __init__(self, inverse=False, **kwargs):
        self.inverse = inverse

    def __call__(self, data, **kwargs):
        img = data['image']
        img_single_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_expanded = np.expand_dims(img_single_channel, 0)

        if self.inverse:
            data['image'] = np.abs(img_expanded - 1)
        else:
            data['image'] = img_expanded

        data['src_image'] = img
        return data
