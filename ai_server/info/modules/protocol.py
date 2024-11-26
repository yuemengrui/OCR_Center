# *_*coding:utf-8 *_*
# @Author : YueMengRui
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, AnyUrl


class ErrorResponse(BaseModel):
    object: str = "Error"
    errcode: int
    errmsg: str


class OCRGeneralRequest(BaseModel):
    image: Optional[str] = Field(default=None,
                                 description="图片base64编码，不包含base64头, 与url二选一，优先级image > url")
    url: Optional[AnyUrl] = Field(default=None, description="图片URL")
    image_direction: Optional[bool] = Field(default=False, description="是否检测图像方向，默认false")
    det_fast: Optional[bool] = Field(default=False, description="是否使用高速检测模型，速度快，默认false")
    rec_fast: Optional[bool] = Field(default=False, description="是否使用高速识别模型，速度快，默认false")
    text_direction: Optional[bool] = Field(default=True, description="是否检测文本行方向, 默认true")
    just_rec: Optional[bool] = Field(default=False, description="是否只识别，不检测")
    drop_score: Optional[float] = Field(default=0.5, ge=0, le=1, description="识别过滤阈值, 取值范围：0～1")
    return_word_box: Optional[bool] = Field(default=False, description="是否返回单字符坐标，默认false")


class OCRResultLine(BaseModel):
    box: List
    text: tuple
    words: List = []


class OCRGeneralResponse(BaseModel):
    object: str = 'OCR General'
    data: List[OCRResultLine]
    time_cost: Dict = {}


class IdcardResponse(BaseModel):
    object: str = 'idcard'
    name: str = ''
    gender: str = ''
    birth: str = ''
    id: str = ''
    address: str = ''
    nation: str = ''


class ImageDirectionRequest(BaseModel):
    image: Optional[str] = Field(default=None,
                                 description="图片base64编码，不包含base64头, 与url二选一，优先级image > url")
    url: Optional[AnyUrl] = Field(default=None, description="图片URL")
    return_correction: Optional[bool] = Field(default=False, description="是否返回纠正后的图像base64，默认false")


class ImageDirectionResponse(BaseModel):
    object: str = 'Image Direction'
    angle: int
    correction_image: str = Field(default=None, description="纠正后的图片base64编码")
    time_cost: Dict = {}


class TableRequest(BaseModel):
    image: Optional[str] = Field(default=None,
                                 description="表格图片base64编码，不包含base64头, 与url二选一，优先级image > url")
    url: Optional[AnyUrl] = Field(default=None, description="表格图片URL")
    with_ocr: Optional[bool] = Field(default=False, description="是否返回表格OCR结果，默认false")
    table_seg_configs: Optional[Dict] = Field(default=dict(), description="可选，表格分割超参数")


class TableAnalysis(BaseModel):
    table_cells: List = []
    ocr: List = []
    html: str = ''


class TableResponse(BaseModel):
    object: str = "Table"
    data: TableAnalysis
    time_cost: Dict = {}


class LayoutRequest(BaseModel):
    image: Optional[str] = Field(default=None,
                                 description="图片base64编码，不包含base64头, 与url二选一，优先级image > url")
    url: Optional[AnyUrl] = Field(default=None, description="图片URL")
    score_threshold: Optional[float] = Field(default=0.3, ge=0, le=1)
    nms_threshold: Optional[float] = Field(default=0.5, ge=0, le=1)


class LayoutOne(BaseModel):
    box: List[int]
    label: str


class LayoutResponse(BaseModel):
    object: str = "Layout"
    data: List[LayoutOne]
    time_cost: Dict = {}
