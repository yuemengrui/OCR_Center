# *_*coding:utf-8 *_*
from fastapi import FastAPI
from . import table


def register_router(app: FastAPI):
    app.include_router(router=table.router, prefix="", tags=["Table"])
