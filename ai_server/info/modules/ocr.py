# *_*coding:utf-8 *_*
# @Author : YueMengRui
import cv2
from mylogger import logger
from fastapi import APIRouter, Request
from info import limiter, text_image_orientation_model, text_det_model, text_rec_model, ocr_model
from configs import API_LIMIT
from .protocol import ErrorResponse, OCRGeneralRequest, OCRResultLine, OCRGeneralResponse, IdcardResponse, \
    ImageDirectionRequest, ImageDirectionResponse
from fastapi.responses import JSONResponse
from info.utils.idcard import idcard_parse
from info.utils.common import request_to_image, resize_4096, cv2_to_base64
from info.utils.response_code import RET, error_map

router = APIRouter()


@router.api_route('/ai/ocr/general', methods=['POST'], response_model=OCRGeneralResponse, summary="General OCR")
@limiter.limit(API_LIMIT['base'])
def ocr_general(request: Request,
                req: OCRGeneralRequest,
                ):
    logger.info(
        {'url': req.url, 'image_direction': req.image_direction, 'det_fast': req.det_fast, 'rec_fast': req.rec_fast,
         'text_direction': req.text_direction, 'drop_score': req.drop_score})

    image = request_to_image(req.image, req.url)

    if image is None:
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(), status_code=500)

    img, scale = resize_4096(image)

    det_model = (text_det_model['mobile'] if req.det_fast and text_det_model['mobile'] is not None else text_det_model[
        'server']) or text_det_model['mobile']

    rec_model = (text_rec_model['mobile'] if req.rec_fast and text_rec_model['mobile'] is not None else text_rec_model[
        'server']) or text_rec_model['mobile']

    res = []
    try:
        boxes, rec_res, time_cost = ocr_model(img=img,
                                              text_detector=det_model,
                                              text_recognizer=rec_model,
                                              cls=req.text_direction,
                                              drop_score=req.drop_score)

        for i in range(len(rec_res)):
            res.append(OCRResultLine(box=[int(x / scale) for x in boxes[i]], text=rec_res[i]))

        return JSONResponse(
            OCRGeneralResponse(data=res, time_cost={k: f"{v:.3f}s" for k, v in time_cost.items()}).dict())
    except Exception as e:
        logger.error({'EXCEPTION': e})
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR]).dict(),
                            status_code=500)


@router.api_route('/ai/ocr/idcard', methods=['POST'], response_model=IdcardResponse, summary="idcard OCR")
@limiter.limit(API_LIMIT['base'])
def ocr_idcard(request: Request,
               req: OCRGeneralRequest,
               ):
    logger.info(
        {'url': req.url, 'image_direction': req.image_direction, 'det_fast': req.det_fast, 'rec_fast': req.rec_fast,
         'text_direction': req.text_direction, 'drop_score': req.drop_score})

    image = request_to_image(req.image, req.url)

    if image is None:
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(), status_code=500)

    img, _ = resize_4096(image)

    det_model = (text_det_model['mobile'] if req.det_fast and text_det_model['mobile'] is not None else text_det_model[
        'server']) or text_det_model['mobile']

    rec_model = (text_rec_model['mobile'] if req.rec_fast and text_rec_model['mobile'] is not None else text_rec_model[
        'server']) or text_rec_model['mobile']

    try:
        _, rec_res, _ = ocr_model(img=img,
                                  text_detector=det_model,
                                  text_recognizer=rec_model,
                                  cls=req.text_direction,
                                  drop_score=req.drop_score)

    except Exception as e:
        logger.error({'EXCEPTION': e})
        rec_res = []

    res = idcard_parse(rec_res)

    return JSONResponse(IdcardResponse(**res).dict())


@router.api_route('/ai/ocr/image_direction', methods=['POST'], response_model=ImageDirectionResponse,
                  summary="image_direction")
@limiter.limit(API_LIMIT['base'])
def image_direction(request: Request,
                    req: ImageDirectionRequest,
                    ):
    image = request_to_image(req.image, req.url)

    if image is None:
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(), status_code=500)

    try:
        cls_res = text_image_orientation_model.predict(image)
        angle = cls_res[0]['label_names'][0]
    except Exception as e:
        logger.error({'EXCEPTION': e})
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR]).dict(),
                            status_code=500)

    base64_str = None
    if req.return_correction:
        cv_rotate_code = {
            '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
            '180': cv2.ROTATE_180,
            '270': cv2.ROTATE_90_CLOCKWISE
        }
        if angle in cv_rotate_code:
            img = cv2.rotate(image, cv_rotate_code[angle])
            base64_str = cv2_to_base64(img)

    return JSONResponse(ImageDirectionResponse(angle=angle, correction_image=base64_str).dict())