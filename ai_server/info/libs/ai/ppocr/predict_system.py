import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import copy
import time
import numpy as np
from .utility import get_rotate_crop_image


class TextSystem(object):
    def __init__(self, text_classifier, logger=None):

        self.logger = logger
        self.text_classifier = text_classifier

    def __call__(self, img, text_detector, text_recognizer, cls=True, drop_score=0.5, return_word_box=False, **kwargs):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            self.logger.debug("no valid image provided")
            return [], [], time_dict

        start = time.time()
        ori_im = img.copy()
        if text_detector is None:
            img_crop_list = [ori_im]
            h, w = ori_im.shape[:2]
            dt_boxes = [np.array([0, 0, w, 0, w, h, h, 0])]
        else:
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

        rec_res, elapse = text_recognizer(img_crop_list, return_word_box=return_word_box)
        time_dict['rec'] = elapse
        self.logger.debug("rec_res num  : {}, elapsed : {}".format(
            len(rec_res), elapse))

        filter_boxes, filter_rec_res, words = [], [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= drop_score:
                temp = []
                for b in box.tolist():
                    temp.extend(b)
                filter_boxes.append(temp)
                filter_rec_res.append((text, score))

                if return_word_box:
                    temp_word_boxes = []
                    word_box_content_list, word_box_list = cal_ocr_word_box(text, box, rec_result[2])
                    word_box_list.sort(key=lambda x: x[0][0])
                    for i in range(len(word_box_content_list)):
                        word_box = []
                        for j in word_box_list[i]:
                            word_box.extend(j)

                        temp_word_boxes.append({'word': word_box_content_list[i], 'box': list(map(int, word_box))})

                    words.append(temp_word_boxes)

        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, words, time_dict


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


def cal_ocr_word_box(rec_str, box, rec_word_info):
    """Calculate the detection frame for each word based on the results of recognition and detection of ocr"""

    col_num, word_list, word_col_list, state_list = rec_word_info
    box = box.tolist()
    bbox_x_start = box[0][0]
    bbox_x_end = box[1][0]
    bbox_y_start = box[0][1]
    bbox_y_end = box[2][1]

    cell_width = (bbox_x_end - bbox_x_start) / col_num

    word_box_list = []
    word_box_content_list = []
    cn_width_list = []
    cn_col_list = []
    for word, word_col, state in zip(word_list, word_col_list, state_list):
        if state == "cn":
            if len(word_col) != 1:
                char_seq_length = (word_col[-1] - word_col[0] + 1) * cell_width
                char_width = char_seq_length / (len(word_col) - 1)
                cn_width_list.append(char_width)
            cn_col_list += word_col
            word_box_content_list += word
        else:
            cell_x_start = bbox_x_start + int(word_col[0] * cell_width)
            cell_x_end = bbox_x_start + int((word_col[-1] + 1) * cell_width)
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)
            word_box_content_list.append("".join(word))
    if len(cn_col_list) != 0:
        if len(cn_width_list) != 0:
            avg_char_width = np.mean(cn_width_list)
        else:
            avg_char_width = (bbox_x_end - bbox_x_start) / len(rec_str)
        for center_idx in cn_col_list:
            center_x = (center_idx + 0.5) * cell_width
            cell_x_start = max(int(center_x - avg_char_width / 2), 0) + bbox_x_start
            cell_x_end = (
                    min(int(center_x + avg_char_width / 2), bbox_x_end - bbox_x_start)
                    + bbox_x_start
            )
            cell = (
                (cell_x_start, bbox_y_start),
                (cell_x_end, bbox_y_start),
                (cell_x_end, bbox_y_end),
                (cell_x_start, bbox_y_end),
            )
            word_box_list.append(cell)

    return word_box_content_list, word_box_list
