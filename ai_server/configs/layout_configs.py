# *_*coding:utf-8 *_*
# @Author : YueMengRui
LAYOUT_CONFIGS = {
    "use_gpu": True,
    "use_mindir": False,
    "gpu_mem": 1024,
    "gpu_id": 0,
    "precision": "fp32",
    "layout_model_dir": "./info/libs/ai/models/picodet_lcnet_x1_0_fgd_layout_cdla_infer",
    "layout_dict_path": "./info/libs/ai/ppstructure/layout_cdla_dict.txt",
    "layout_score_threshold": 0.3,
    "layout_nms_threshold": 0.5
}
