# *_*coding:utf-8 *_*
# @Author : YueMengRui
from mylogger import logger
from fastapi import APIRouter, Request
from info import limiter, layout_model
from configs import API_LIMIT
from .protocol import ErrorResponse, LayoutRequest, LayoutResponse, LayoutOne
from fastapi.responses import JSONResponse
from info.utils.response_code import RET, error_map
from info.utils.common import request_to_image

router = APIRouter()


@router.api_route('/layout/analysis', methods=['POST'], response_model=LayoutResponse, summary="Layout Analysis")
@limiter.limit(API_LIMIT['layout'])
def layout_analysis(request: Request,
                    req: LayoutRequest,
                    ):
    logger.info({'url': req.url})

    image = request_to_image(req.image, req.url)

    if image is None:
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(), status_code=500)

    try:
        res, time_cost = layout_model(img=image, score_threshold=req.score_threshold, nms_threshold=req.nms_threshold)
        return JSONResponse(LayoutResponse(data=[LayoutOne(**x) for x in res],
                                           time_cost={k: f"{v:.3f}s" for k, v in time_cost.items()}).dict())
    except Exception as e:
        logger.error({'EXCEPTION': e})
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR]).dict(),
                            status_code=500)
