import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import numpy as np
import time

import info.libs.ai.ppocr.utility as utility
from info.libs.ai.ppocr.data import create_operators, transform
from info.libs.ai.ppocr.postprocess import build_post_process


class LayoutPredictor(object):
    def __init__(self, args, logger=None):
        pre_process_list = [{
            'Resize': {
                'size': [800, 608]
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image']
            }
        }]
        postprocess_params = {
            'name': 'PicoDetPostProcess',
            "layout_dict_path": args.layout_dict_path,
            "score_threshold": args.layout_score_threshold,
            "nms_threshold": args.layout_nms_threshold,
        }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'layout', logger)

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]

        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        img = img.copy()

        starttime = time.time()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()

        np_score_list, np_boxes_list = [], []
        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)
        for out_idx in range(num_outs):
            np_score_list.append(
                self.predictor.get_output_handle(output_names[out_idx])
                .copy_to_cpu())
            np_boxes_list.append(
                self.predictor.get_output_handle(output_names[
                                                     out_idx + num_outs]).copy_to_cpu())
        preds = dict(boxes=np_score_list, boxes_num=np_boxes_list)

        post_preds = self.postprocess_op(ori_im, img, preds)
        res = [{'box': [int(x) for x in i['bbox'].tolist()], 'label': i['label']} for i in post_preds]
        elapse = time.time() - starttime
        return res, {'layout': elapse}
