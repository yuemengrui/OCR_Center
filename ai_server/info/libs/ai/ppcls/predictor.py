import platform
import os
from paddle.inference import Config
from paddle.inference import create_predictor


class Predictor(object):
    def __init__(self, args, inference_model_dir=None):
        # HALF precission predict only work when using tensorrt
        if args.use_fp16 is True:
            assert args.use_tensorrt is True
        self.args = args

        self.predictor, self.config = self.create_paddle_predictor(
            args, inference_model_dir)

    def predict(self, image):
        raise NotImplementedError

    def create_paddle_predictor(self, args, inference_model_dir=None):
        if inference_model_dir is None:
            inference_model_dir = args.inference_model_dir
        if "inference_int8.pdiparams" in os.listdir(inference_model_dir):
            params_file = os.path.join(inference_model_dir,
                                       "inference_int8.pdiparams")
            model_file = os.path.join(inference_model_dir,
                                      "inference_int8.pdmodel")
            assert args.get(
                "use_fp16", False
            ) is False, "fp16 mode is not supported for int8 model inference, please set use_fp16 as False during inference."
        else:
            params_file = os.path.join(inference_model_dir,
                                       "inference.pdiparams")
            model_file = os.path.join(inference_model_dir, "inference.pdmodel")
            assert args.get(
                "use_int8", False
            ) is False, "int8 mode is not supported for fp32 model inference, please set use_int8 as False during inference."

        config = Config(model_file, params_file)

        if args.get("use_gpu", False):
            config.enable_use_gpu(args.gpu_mem, 0)
        elif args.get("use_npu", False):
            config.enable_custom_device('npu')
        elif args.get("use_xpu", False):
            config.enable_xpu()
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                # there is no set_mkldnn_cache_capatity() on macOS
                if platform.system() != "Darwin":
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(args.cpu_num_threads)

        if args.enable_profile:
            config.enable_profile()
        config.disable_glog_info()
        config.switch_ir_optim(args.ir_optim)  # default true
        if args.use_tensorrt:
            precision = Config.Precision.Float32
            if args.get("use_int8", False):
                precision = Config.Precision.Int8
            elif args.get("use_fp16", False):
                precision = Config.Precision.Half

            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=args.batch_size,
                workspace_size=1 << 30,
                min_subgraph_size=30,
                use_calib_mode=False)

        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        return predictor, config
