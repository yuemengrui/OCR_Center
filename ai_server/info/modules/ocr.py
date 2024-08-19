# *_*coding:utf-8 *_*
# @Author : YueMengRui
import cv2
from mylogger import logger
from fastapi import APIRouter, Request
from info import limiter, text_image_orientation_model, text_det_model, text_rec_model, ocr_model, layout_model, \
    table_model
from configs import API_LIMIT
from .protocol import ErrorResponse, OCRGeneralRequest, OCRResultLine, OCRGeneralResponse, IdcardResponse, \
    ImageDirectionRequest, ImageDirectionResponse
from fastapi.responses import JSONResponse
from info.utils.idcard import idcard_parse
from info.utils.common import request_to_image, resize_4096, cv2_to_base64, small_h_image_handle
from info.utils.response_code import RET, error_map

router = APIRouter()


@router.api_route('/general', methods=['POST'], response_model=OCRGeneralResponse, summary="General OCR")
@limiter.limit(API_LIMIT['ocr'])
def ocr_general(request: Request,
                req: OCRGeneralRequest,
                ):
    logger.info(
        {'url': req.url, 'image_direction': req.image_direction, 'det_fast': req.det_fast, 'rec_fast': req.rec_fast,
         'text_direction': req.text_direction, 'drop_score': req.drop_score, 'return_word_box': req.return_word_box})

    image = request_to_image(req.image, req.url)

    if image is None:
        return JSONResponse(ErrorResponse(errcode=RET.PARAMERR, errmsg=error_map[RET.PARAMERR]).dict(), status_code=500)

    image = small_h_image_handle(image)
    img, scale = resize_4096(image)

    if req.just_rec:
        det_model = None
    else:
        det_model = (text_det_model['mobile'] if req.det_fast and text_det_model['mobile'] is not None else text_det_model[
            'server']) or text_det_model['mobile']

    rec_model = (text_rec_model['mobile'] if req.rec_fast and text_rec_model['mobile'] is not None else text_rec_model[
        'server']) or text_rec_model['mobile']

    res = []
    try:
        boxes, rec_res, words, time_cost = ocr_model(img=img,
                                                     text_detector=det_model,
                                                     text_recognizer=rec_model,
                                                     cls=req.text_direction,
                                                     drop_score=req.drop_score,
                                                     return_word_box=req.return_word_box)

        for i in range(len(rec_res)):
            if req.return_word_box:
                res.append(OCRResultLine(box=[int(x / scale) for x in boxes[i]], text=rec_res[i], words=words[i]))
            else:
                res.append(OCRResultLine(box=[int(x / scale) for x in boxes[i]], text=rec_res[i]))

        return JSONResponse(
            OCRGeneralResponse(data=res, time_cost={k: f"{v:.3f}s" for k, v in time_cost.items()}).dict())
    except Exception as e:
        logger.error({'EXCEPTION': e})
        return JSONResponse(ErrorResponse(errcode=RET.SERVERERR, errmsg=error_map[RET.SERVERERR]).dict(),
                            status_code=500)


@router.api_route('/idcard', methods=['POST'], response_model=IdcardResponse, summary="idcard OCR")
@limiter.limit(API_LIMIT['ocr'])
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
        _, rec_res, _, _ = ocr_model(img=img,
                                     text_detector=det_model,
                                     text_recognizer=rec_model,
                                     cls=req.text_direction,
                                     drop_score=req.drop_score)

    except Exception as e:
        logger.error({'EXCEPTION': e})
        rec_res = []

    res = idcard_parse(rec_res)

    return JSONResponse(IdcardResponse(**res).dict())


@router.api_route('/image_direction', methods=['POST'], response_model=ImageDirectionResponse,
                  summary="image_direction")
@limiter.limit(API_LIMIT['image_direction'])
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

    base64_str = ''
    if req.return_correction:
        cv_rotate_code = {
            '90': cv2.ROTATE_90_COUNTERCLOCKWISE,
            '180': cv2.ROTATE_180,
            '270': cv2.ROTATE_90_CLOCKWISE
        }
        if angle in cv_rotate_code:
            image = cv2.rotate(image, cv_rotate_code[angle])

        base64_str = cv2_to_base64(image)

    return JSONResponse(ImageDirectionResponse(angle=angle, correction_image=base64_str).dict())


@router.api_route('/test', methods=['GET'])
@limiter.limit("6/minute")
def ocr_server_test(request: Request):
    image = cv2.imread('static/ocr.png')
    table_image = cv2.imread('static/table.jpg')

    msg = ''
    try:
        res = table_model(img=table_image)
        logger.info(f"table: {res}")
        msg += 'table model is OK! \n '
    except Exception as e:
        logger.error({'EXCEPTION': e})
        msg += 'table model exception!!! \n '

    try:
        res = layout_model(img=image)
        logger.info(f"layout: {res}")
        msg += 'layout model is OK! \n '
    except Exception as e:
        logger.error({'EXCEPTION': e})
        msg += 'layout model exception!!! \n '

    try:
        res = text_image_orientation_model.predict(image)
        logger.info(f"image_orientation: {res}")
        msg += 'image_orientation is OK! \n '
    except Exception as e:
        logger.error({'EXCEPTION': e})
        msg += 'image_orientation exception!!! \n '

    if text_det_model['server'] is not None:
        try:
            res = text_det_model['server'](image)
            logger.info(f"det server: {res}")
            msg += 'det server model is OK! \n '
        except Exception as e:
            logger.error({'EXCEPTION': e})
            msg += 'det server model exception!!! \n '

    if text_det_model['mobile'] is not None:
        try:
            res = text_det_model['mobile'](image)
            logger.info(f"det mobile: {res}")
            msg += 'det mobile model is OK! \n '
        except Exception as e:
            logger.error({'EXCEPTION': e})
            msg += 'det mobile model exception!!! \n '

    if text_rec_model['server'] is not None:
        try:
            res = text_rec_model['server']([image])
            logger.info(f"rec server: {res}")
            msg += 'rec server model is OK! \n '
        except Exception as e:
            logger.error({'EXCEPTION': e})
            msg += 'rec server model exception!!! \n '

    if text_rec_model['mobile'] is not None:
        try:
            res = text_rec_model['mobile']([image])
            logger.info(f"rec mobile: {res}")
            msg += 'rec mobile model is OK! \n '
        except Exception as e:
            logger.error({'EXCEPTION': e})
            msg += 'rec mobile model exception!!! \n '

    try:
        _, res, _, _ = ocr_model(img=image,
                                 text_detector=text_det_model['server'] or text_det_model['mobile'],
                                 text_recognizer=text_rec_model['server'] or text_rec_model['mobile'],
                                 )
        res = [x[0] for x in res]
        text = '\n'.join(res)
        logger.info(f"ocr: {text}")
        msg += 'ocr model is OK! \n '
        msg += text
    except Exception as e:
        logger.error({'EXCEPTION': e})
        msg += 'ocr model exception!!! \n '

    return JSONResponse({'msg': msg})
