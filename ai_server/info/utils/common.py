# *_*coding:utf-8 *_*
# @Author : YueMengRui
import cv2
import base64
import requests
import numpy as np
from mylogger import logger


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


def cv2_to_base64(image):
    return base64.b64encode(np.array(cv2.imencode('.jpg', image)[1]).tobytes()).decode('utf-8')


def base64_to_cv2(b64str: str):
    data = base64.b64decode(b64str.encode('utf-8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def bytes_to_cv2(data: bytes):
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def request_to_image(image: str, url: str):
    if image:
        try:
            return base64_to_cv2(image)
        except Exception as e:
            logger.error({'EXCEPTION': e})

    try:
        return bytes_to_cv2(requests.get(url).content)
    except Exception as e:
        logger.error({'EXCEPTION': e})

    return None


def small_h_image_handle(image):
    h, w = image.shape[:2]

    if h < 32:
        scale = 32 / h
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)

    if h < 64:
        k = int((64 - h) / 2)
        image = cv2.copyMakeBorder(image, k, k, 1, 1, cv2.BORDER_REPLICATE)

    return image


def resize_4096(image):
    scale = 1
    h, w = image.shape[:2]

    if max(h, w) > 4096:
        scale = 4096 / max(h, w)
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)

    return image, scale
