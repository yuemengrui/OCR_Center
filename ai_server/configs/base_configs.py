# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

FASTAPI_TITLE = 'OCR_Center'
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 24666

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
TEMP = './temp'
os.makedirs(TEMP, exist_ok=True)

DET_LOAD_SERVER = True
DET_LOAD_MOBILE = True
REC_LOAD_SERVER = True
REC_LOAD_MOBILE = True

# API LIMIT
API_LIMIT = {
    "ocr": "1000/minute",
    "layout": "1000/minute",
    "table": "1000/minute",
    "image_direction": "1000/minute"
}
