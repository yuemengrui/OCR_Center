# *_*coding:utf-8 *_*
from fastapi import FastAPI
from . import table, ocr, layout


def register_router(app: FastAPI):
    app.include_router(router=ocr.router, prefix="/ai/ocr", tags=["OCR"])
    app.include_router(router=table.router, prefix="/ai/ocr", tags=["Table"])
    app.include_router(router=layout.router, prefix="/ai/ocr", tags=["Layout"])
