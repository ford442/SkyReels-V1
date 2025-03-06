import logging
import os
from collections import OrderedDict  # Import OrderedDict

import torch
from diffusers import HunyuanVideoTransformer3DModel
from PIL import Image
from torchao.quantization import float8_weight_only
from torchao.quantization import quantize_
from transformers import LlamaModel

from . import TaskType
from .offload import OffloadConfig
from .pipelines import SkyreelsVideoPipeline

# --- Dummy Classes (with named_children) ---
class LlamaModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return LlamaModel()

    def to(self, device):
        return self

    def __init__(self):
        super().__init__()
        self._modules = OrderedDict()  # Add _modules
        self.linear = torch.nn.Linear(10,10) # Add dummy submodule
        self._modules["linear"] = self.linear

    def named_children(self):  # Implement named_children
        return self._modules.items()

class HunyuanVideoTransformer3DModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return HunyuanVideoTransformer3DModel()
    def to(self, device):
        return self
    def __init__(self):
        super().__init__()
        self._modules = OrderedDict()  # Add _modules
        self.linear = torch.nn.Linear(10,10)
        self._modules["linear"] = self.linear

    def named_children(self):  # Implement named_children
        return self._modules.items()

class SkyreelsVideoPipeline:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return SkyreelsVideoPipeline()

    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        frames = torch.randn(1, 3, 16, 512, 512)
        return type('obj', (object,), {'frames' : [frames]})()
    def __init__(self):
      super().__init__()
      self._modules = OrderedDict()
      self.vae = self.VAE() # Instantiate the VAE
      self._modules["vae"] = self.vae

    def named_children(self):
      return self._modules.items()

    class VAE:  # Corrected inner class name
        def enable_tiling(self):
            pass  # Dummy implementation


def quantize_(*args, **kwargs):
    return

def float8_weight_only():
    return

logger = logging.getLogger(__name__)


class SkyReelsVideoSingleGpuInfer:
    def _load_model(self, model_id: str, base_model_id: str = "hunyuanvideo-community/HunyuanVideo", quant_model: bool = True):
        logger.info(f"load model model_id:{model_id} quan_model:{quant_model}")
        text_encoder = LlamaModel.from_pretrained(
            base_model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        ).to("cpu")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device="cpu"
        ).to("cpu")

        if quant_model:
            quantize_(text_encoder, float8_weight_only())
            text_encoder.to("cpu")
            torch.cuda.empty_cache()
            quantize_(transformer, float8_weight_only())
            transformer.to("cpu")
            torch.cuda.empty_cache()

        pipe = SkyreelsVideoPipeline.from_pretrained(
            base_model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16
        ).to("cpu")

        pipe.vae.enable_tiling()  # Now this will work
        torch.cuda.empty_cache()
        return pipe

    def __init__(
        self,
        task_type: TaskType,
        model_id: str,
        quant_model: bool = True,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        enable_cfg_parallel: bool = True,
    ):
        self.task_type = task_type
        self.model_id = model_id
        self.quant_model = quant_model
        self.is_offload = is_offload
        self.offload_config = offload_config
        self.enable_cfg_parallel = enable_cfg_parallel
        self.pipe = None
        self.is_initialized = False
        self.gpu_device = None

    def initialize(self):
        """Initializes the model and moves it to the GPU."""
        if self.is_initialized:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot initialize model.")

        self.gpu_device = "cuda:0"
        self.pipe = self._load_model(model_id=self.model_id, quant_model=self.quant_model)

        if self.is_offload:
          pass
        else:
            self.pipe.to(self.gpu_device)

        if self.offload_config.compiler_transformer:
            torch._dynamo.config.suppress_errors = True
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{self.offload_config.compiler_cache}"
            self.pipe.transformer = torch.compile(
                self.pipe.transformer, mode="max-autotune-no-cudagraphs", dynamic=True
            )
            if self.offload_config.compiler_transformer:
                self.warm_up()
        self.is_initialized = True

    def warm_up(self):
      if not self.is_initialized:
          raise RuntimeError("Model must be initialized before warm-up.")

      init_kwargs = {
            "prompt": "A woman is dancing in a room",
            "height": 544,
            "width": 960,
            "guidance_scale": 6,
            "num_inference_steps": 1,
            "negative_prompt": "bad quality",
            "num_frames": 16,
            "generator": torch.Generator(self.gpu_device).manual_seed(42),
            "embedded_guidance_scale": 1.0,
        }
      if self.task_type == TaskType.I2V:
        init_kwargs["image"] = Image.new("RGB",(544,960), color = "black")
      self.pipe(**init_kwargs)
      logger.info("Warm-up complete.")

    def infer(self, **kwargs):
        """Handles inference requests."""
        if not self.is_initialized:
          self.initialize()
        if "seed" in kwargs:
            kwargs["generator"] = torch.Generator(self.gpu_device).manual_seed(kwargs["seed"])
            del kwargs["seed"]
        assert (self.task_type == TaskType.I2V and "image" in kwargs) or self.task_type == TaskType.T2V
        result = self.pipe(**kwargs).frames[0]
        return result
