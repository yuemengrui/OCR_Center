# *_*coding:utf-8 *_*
# @Author : YueMengRui
REC_CONFIGS = {
    "global": {
        "use_gpu": True,
        "gpu_mem": 1024,
        "gpu_id": 0,
        "precision": "fp32",
        "rec_algorithm": "SVTR_LCNet",
        "rec_image_inverse": True,
        "rec_image_shape": "3, 48, 320",
        "rec_batch_num": 6,
        "max_text_length": 32,
        "rec_char_dict_path": "./info/libs/ai/ppocr/ppocr_keys_v1.txt",
        "use_space_char": True,
        "drop_score": 0.5
    },
    "server": {
        "rec_model_dir": "./info/libs/ai/models/ch_PP-OCRv4_rec_server_infer"
    },
    "mobile": {
        "rec_model_dir": "./info/libs/ai/models/ch_PP-OCRv4_rec_infer"
    }
}
