import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import copy
import time
from .utility import get_rotate_crop_image


class TextSystem(object):
    def __init__(self, text_classifier, logger=None):

        self.logger = logger
        self.text_classifier = text_classifier

    def __call__(self, img, text_detector, text_recognizer, cls=True, drop_score=0.5, **kwargs):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            self.logger.debug("no valid image provided")
            return [], [], time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            self.logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return [], [], time_dict
        else:
            self.logger.debug("dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            self.logger.debug("cls num  : {}, elapsed : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        self.logger.debug("rec_res num  : {}, elapsed : {}".format(
            len(rec_res), elapse))

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= drop_score:
                temp = []
                for b in box.tolist():
                    temp.extend(b)
                filter_boxes.append(temp)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes
