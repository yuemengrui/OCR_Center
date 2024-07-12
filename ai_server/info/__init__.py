# *_*coding:utf-8 *_*
import time
from configs import DET_LOAD_SERVER, DET_LOAD_MOBILE, REC_LOAD_MOBILE, REC_LOAD_SERVER, DET_CONFIGS, REC_CONFIGS, \
    CLS_CONFIGS, LAYOUT_CONFIGS, TABLE_CONFIGS
from mylogger import logger
from fastapi.requests import Request
from starlette.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from info.utils.common import DotDict
from info.libs.ai.ppcls.predict_cls import ClsPredictor
from info.libs.ai.ppocr.predict_det import TextDetector
from info.libs.ai.ppocr.predict_rec import TextRecognizer
from info.libs.ai.ppocr.predict_cls import TextClassifier
from info.libs.ai.ppocr.predict_system import TextSystem
from info.libs.ai.ppstructure.layout.predict_layout import LayoutPredictor
from info.libs.ai.ppstructure.table.predict_table import TableSystem

####################### load model #######################
text_image_orientation_model = ClsPredictor()
logger.info('text_image_orientation_model load successful!')
text_cls_model = TextClassifier(args=DotDict(CLS_CONFIGS), logger=logger)
logger.info('text_cls_model load successful!')

text_det_model = {
    'server': None,
    'mobile': None
}
if DET_LOAD_SERVER == False and DET_LOAD_MOBILE == False:
    DET_LOAD_SERVER = True
if DET_LOAD_SERVER:
    det_server_configs = DET_CONFIGS['server']
    det_server_configs.update(DET_CONFIGS['global'])
    text_det_model['server'] = TextDetector(args=DotDict(det_server_configs), logger=logger)
    logger.info('text det model server version load successful!')
if DET_LOAD_MOBILE:
    det_mobile_configs = DET_CONFIGS['mobile']
    det_mobile_configs.update(DET_CONFIGS['global'])
    text_det_model['mobile'] = TextDetector(args=DotDict(det_mobile_configs), logger=logger)
    logger.info('text det model mobile version load successful!')

text_rec_model = {
    'server': None,
    'mobile': None
}
if REC_LOAD_SERVER == False and REC_LOAD_MOBILE == False:
    REC_LOAD_SERVER = True
if REC_LOAD_SERVER:
    rec_server_configs = REC_CONFIGS['server']
    rec_server_configs.update(REC_CONFIGS['global'])
    text_rec_model['server'] = TextRecognizer(args=DotDict(rec_server_configs), logger=logger)
    logger.info('text rec model server version load successful!')
if REC_LOAD_MOBILE:
    rec_mobile_configs = REC_CONFIGS['mobile']
    rec_mobile_configs.update(REC_CONFIGS['global'])
    text_rec_model['mobile'] = TextRecognizer(args=DotDict(rec_mobile_configs), logger=logger)
    logger.info('text rec model mobile version load successful!')

ocr_model = TextSystem(text_classifier=text_cls_model, logger=logger)

layout_model = LayoutPredictor(args=DotDict(LAYOUT_CONFIGS), logger=logger)
logger.info('layout_model load successful!')
table_model = TableSystem(args=DotDict(TABLE_CONFIGS), logger=logger)
logger.info('table_model load successful!')

####################### load model #######################

limiter = Limiter(key_func=lambda *args, **kwargs: '127.0.0.1')


def app_registry(app):
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.middleware("http")
    async def api_time_cost(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        cost = time.time() - start
        logger.info(f'end request "{request.method} {request.url.path}" - {cost:.3f}s')
        return response

    app.mount("/ai/ocr/static", StaticFiles(directory=f"static"), name="static")

    @app.get("/ai/ocr/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="/ai/ocr/static/swagger-ui-bundle.js",
            swagger_css_url="/ai/ocr/static/swagger-ui.css",
        )

    @app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/ai/ocr/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " - ReDoc",
            redoc_js_url="/ai/ocr/static/redoc.standalone.js",
        )

    from info.modules import register_router

    register_router(app)
