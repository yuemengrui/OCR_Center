# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


def build_pre_process_list(args):
    resize_op = {'ResizeTableImage': {'max_len': args.table_max_len, }}
    pad_op = {
        'PaddingTableImage': {
            'size': [args.table_max_len, args.table_max_len]
        }
    }
    normalize_op = {
        'NormalizeImage': {
            'std': [0.229, 0.224, 0.225] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'mean': [0.485, 0.456, 0.406] if
            args.table_algorithm not in ['TableMaster'] else [0.5, 0.5, 0.5],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }
    to_chw_op = {'ToCHWImage': None}
    keep_keys_op = {'KeepKeys': {'keep_keys': ['image', 'shape']}}
    if args.table_algorithm not in ['TableMaster']:
        pre_process_list = [
            resize_op, normalize_op, pad_op, to_chw_op, keep_keys_op
        ]
    else:
        pre_process_list = [
            resize_op, pad_op, normalize_op, to_chw_op, keep_keys_op
        ]
    return pre_process_list


class TableStructurer(object):
    def __init__(self, args, logger=None):
        self.args = args

        pre_process_list = build_pre_process_list(args)
        if args.table_algorithm not in ['TableMaster']:
            postprocess_params = {
                'name': 'TableLabelDecode',
                "character_dict_path": args.table_char_dict_path,
                'merge_no_span_structure': args.merge_no_span_structure
            }
        else:
            postprocess_params = {
                'name': 'TableMasterLabelDecode',
                "character_dict_path": args.table_char_dict_path,
                'box_shape': 'pad',
                'merge_no_span_structure': args.merge_no_span_structure
            }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'table', logger)

    def __call__(self, img):
        starttime = time.time()

        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        preds['structure_probs'] = outputs[1]
        preds['loc_preds'] = outputs[0]

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        structure_str_list = post_result['structure_batch_list'][0]
        bbox_list = post_result['bbox_batch_list'][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = [
                                 '<html>', '<body>', '<table>'
                             ] + structure_str_list + ['</table>', '</body>', '</html>']
        elapse = time.time() - starttime

        return (structure_str_list, bbox_list), elapse
