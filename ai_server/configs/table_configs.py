# *_*coding:utf-8 *_*
# @Author : YueMengRui
TABLE_CONFIGS = {
    "use_gpu": True,
    "gpu_mem": 1024,
    "gpu_id": 0,
    "precision": "fp32",
    "table_max_len": 488,
    "table_algorithm": "TableAttn",
    "table_model_dir": "./info/libs/ai/models/ch_ppstructure_mobile_v2.0_SLANet_infer",
    "merge_no_span_structure": True,
    "table_char_dict_path": "./info/libs/ai/ppstructure/table_structure_dict_ch.txt"
}
