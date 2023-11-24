# *_*coding:utf-8 *_*
# @Author : YueMengRui
DET_CONFIGS = {
    "global": {
        "use_gpu": True,
        "gpu_mem": 1024,
        "gpu_id": 0,
        "precision": "fp32",
        "det_algorithm": "DB",
        "det_limit_side_len": 960,
        "det_limit_type": "max",
        "det_box_type": "quad",
        "det_db_thresh": 0.3,
        "det_db_box_thresh": 0.6,
        "det_db_unclip_ratio": 1.5,
        "max_batch_size": 10,
        "use_dilation": False,
        "det_db_score_mode": "fast",
    },
    "server": {
        "det_model_dir": "./info/libs/ai/models/ch_PP-OCRv4_det_server_infer"
    },
    "mobile": {
        "det_model_dir": "./info/libs/ai/models/ch_PP-OCRv4_det_infer"
    }

}
