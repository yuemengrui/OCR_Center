# *_*coding:utf-8 *_*
# @Author : YueMengRui
from mylogger import logger
from fastapi import APIRouter, Request
from info import limiter, table_model, text_det_model, text_rec_model
from configs import API_LIMIT
from .protocol import TableRequest, ErrorResponse, TableResponse, TableAnalysis
from fastapi.responses import JSONResponse
from info.utils.response_code import RET, error_map
from info.utils.common import request_to_image

router = APIRouter()


@router.api_route('/ai/table/analysis', methods=['POST'], response_model=TableResponse, summary="Table Analysis")
@limiter.limit(API_LIMIT['base'])
def table_ocr(request: Request,
              req: TableRequest,
              ):
    logger.info(
        {'url': req.url, 'redraw': req.redraw, 'table_seg_configs': req.table_seg_configs, 'with_ocr': req.with_ocr})

    image = request_to_image(req.image, req.url)

    if image is None:
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(), status_code=500)

    text_detector = None
    text_recognizer = None
    if req.with_ocr:
        text_detector = text_det_model['server'] or text_det_model['mobile']
        text_recognizer = text_rec_model['server'] or text_rec_model['mobile']

    try:
        res, time_cost = table_model(img=image, ocr=req.with_ocr, text_detector=text_detector,
                                     text_recognizer=text_recognizer)
        return JSONResponse(
            TableResponse(data=TableAnalysis(**res), time_cost={k: f"{v:.3f}s" for k, v in time_cost.items()}).dict())
    except Exception as e:
        logger.error({'EXCEPTION': e})
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR]).dict(),
                            status_code=500)
