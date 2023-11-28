import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import copy
import numpy as np
import time
import info.libs.ai.ppocr.utility as utility
from info.libs.ai.ppocr.predict_system import sorted_boxes
from info.libs.ai.ppstructure.table.matcher import TableMatch
from info.libs.ai.ppstructure.table.table_master_match import TableMasterMatcher
import info.libs.ai.ppstructure.table.predict_structure as predict_strture


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    #     print(shape)
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


class TableSystem(object):
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger

        self.table_structurer = predict_strture.TableStructurer(args, logger=self.logger)
        if args.table_algorithm in ['TableMaster']:
            self.match = TableMasterMatcher()
        else:
            self.match = TableMatch(filter_ocr_result=True)

        self.predictor, self.input_tensor, self.output_tensors, self.config = utility.create_predictor(
            args, 'table', logger)

    def __call__(self, img, ocr=False, text_detector=None, text_recognizer=None, **kwargs):
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()
        structure_res, elapse = self._structure(copy.deepcopy(img))
        result['table_cells'] = [list(map(int, x)) for x in structure_res[1].tolist()]
        time_dict['table'] = elapse

        if ocr and text_detector and text_recognizer:
            dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(
                copy.deepcopy(img), text_detector, text_recognizer)
            time_dict['det'] = det_elapse
            time_dict['rec'] = rec_elapse

            dt_boxes = [list(map(int, x.tolist())) for x in dt_boxes]
            result['ocr'] = [{'box': dt_boxes[i], 'text': rec_res[i]} for i in range(len(rec_res))]

            tic = time.time()
            pred_html = self.match(structure_res, dt_boxes, rec_res)
            toc = time.time()
            time_dict['match'] = toc - tic
            result['html'] = pred_html

        end = time.time()
        time_dict['all'] = end - start
        return result, time_dict

    def _structure(self, img):
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        return structure_res, elapse

    def _ocr(self, img, text_detector, text_recognizer):
        h, w = img.shape[:2]
        dt_boxes, det_elapse = text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        self.logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), det_elapse))
        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0):int(y1), int(x0):int(x1), :]
            img_crop_list.append(text_rect)
        rec_res, rec_elapse = text_recognizer(img_crop_list)
        self.logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), rec_elapse))
        return dt_boxes, rec_res, det_elapse, rec_elapse
