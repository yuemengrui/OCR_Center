# *_*coding:utf-8 *_*
# @Author : YueMengRui
CLS_CONFIGS = {
    "use_gpu": True,
    "use_mindir": False,
    "gpu_mem": 256,
    "gpu_id": 0,
    "precision": "fp32",
    "cls_model_dir": "./info/libs/ai/models/ch_ppocr_mobile_v2.0_cls_infer",
    "cls_image_shape": "3, 48, 192",
    "label_list": ['0', '180'],
    "cls_batch_num": 6,
    "cls_thresh": 0.9,
    "warmup": True,
}
