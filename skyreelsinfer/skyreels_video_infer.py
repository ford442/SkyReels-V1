import logging
import os
import threading
import time
from datetime import timedelta
from typing import Any
from typing import Dict
import contextlib

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import HunyuanVideoTransformer3DModel
from PIL import Image
from torchao.quantization import float8_weight_only
from torchao.quantization import quantize_
from transformers import LlamaModel

from . import TaskType
from .offload import Offload
from .offload import OffloadConfig
from .pipelines import SkyreelsVideoPipeline

@contextlib.contextmanager
def spawn_context():
    ctx = mp.get_context("spawn")
    yield ctx
    
class OffloadConfig:
    def __init__(self, high_cpu_memory=False, parameters_level=False, compiler_transformer=False, compiler_cache=""):
        self.high_cpu_memory = high_cpu_memory
        self.parameters_level = parameters_level
        self.compiler_transformer = compiler_transformer
        self.compiler_cache = compiler_cache

class TaskType:
  T2V = 0
  I2V = 1
class LlamaModel: # Dummy model for testing
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return LlamaModel()
    def to(self, device):
        return self
class HunyuanVideoTransformer3DModel: #Dummy model
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return HunyuanVideoTransformer3DModel()
    def to(self, device):
        return self
class SkyreelsVideoPipeline: #Dummy class
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return SkyreelsVideoPipeline()
    def to(self, device):
        return self
    def __call__(self, *args, **kwargs):
        return self
    class vae:
        @staticmethod
        def enable_tiling():
            return
def quantize_(*args, **kwargs): #Dummy Function
    return
def float8_weight_only(): #Dummy function
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
        pipe.vae.enable_tiling()
        torch.cuda.empty_cache()
        return pipe

    def __init__(
        self,
        task_type: TaskType,
        model_id: str,
        quant_model: bool = True,
        local_rank: int = 0,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        enable_cfg_parallel: bool = True,
    ):
        # Store configuration only.  NO GPU or process setup here.
        self.task_type = task_type
        self.gpu_rank = local_rank
        self.model_id = model_id
        self.quant_model = quant_model
        self.world_size = world_size
        self.is_offload = is_offload
        self.offload_config = offload_config
        self.enable_cfg_parallel = enable_cfg_parallel
        self.pipe = None
        self.is_initialized = False
        self.gpu_device = None

    def initialize(self):
        """Initializes the model (but NOT the process)."""
        if self.is_initialized:
            return  # Already initialized

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot initialize model.")

        self.gpu_device = f"cuda:{self.gpu_rank}"
        self.pipe = self._load_model(model_id=self.model_id, quant_model=self.quant_model)

        max_batch_dim_size = 2 if self.enable_cfg_parallel and self.world_size > 1 else 1
        logger.info(f"max_batch_dim_size: {max_batch_dim_size}")
        if self.is_offload:
            pass # Offload call
        else:
            self.pipe.to(self.gpu_device)

        if self.offload_config.compiler_transformer:
            torch._dynamo.config.suppress_errors = True
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{self.offload_config.compiler_cache}_{self.world_size}"
            self.pipe.transformer = torch.compile(
                self.pipe.transformer, mode="max-autotune-no-cudagraphs", dynamic=True
            )
            # warm_up is now separate.

        self.is_initialized = True

    def warm_up(self):
        """Performs a warm-up inference (optional)."""
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
          init_kwargs["image"] = Image.new("RGB",(544,960), color = "black") #Dummy black image for init
        self.pipe(**init_kwargs)
        logger.info("Warm-up complete.")

    def inference(self, request_queue: mp.Queue, response_queue: mp.Queue):
      """Handles inference requests in the child process."""

      # Signal readiness *before* waiting for INIT
      response_queue.put("READY")
      logger.info(f"rank:{self.gpu_rank} waiting for initialization request")

      # Wait for an initialization signal
      init_signal = request_queue.get()
      if init_signal == "INIT":
          self.initialize()  # Initialize the model
          logger.info(f"rank:{self.gpu_rank} finish init pipe")
          response_queue.put("INITIALIZED")
      else:
          error_msg = f"rank: {self.gpu_rank} received unexpected message: {init_signal}"
          logger.error(error_msg)
          response_queue.put(RuntimeError(error_msg))
          return  # Exit on error

      # Wait for a warmup signal
      warmup_signal = request_queue.get()
      if warmup_signal == "WARMUP":
        if self.offload_config.compiler_transformer:
          self.warm_up()  # Perform warm-up if needed
        response_queue.put("WARMUP_COMPLETE")
      else:
          error_msg = f"rank: {self.gpu_rank} received unexpected message: {warmup_signal}"
          logger.error(error_msg)
          response_queue.put(RuntimeError(error_msg))
          return  # Exit on error

      # Main inference loop
      while True:
          logger.info(f"rank:{self.gpu_rank} waiting for request")
          kwargs = request_queue.get()  # Wait for a request
          logger.info(f"rank:{self.gpu_rank} kwargs: {kwargs}")

          if "seed" in kwargs:
              kwargs["generator"] = torch.Generator(self.gpu_device).manual_seed(kwargs["seed"])
              del kwargs["seed"]  # Remove seed, as it's now in the generator

          start_time = time.time()
          try:
              assert (self.task_type == TaskType.I2V and "image" in kwargs) or self.task_type == TaskType.T2V
              out = self.pipe(**kwargs).frames[0]  # Run inference
              logger.info(f"rank:{self.gpu_rank} inference time: {time.time() - start_time}")
              response_queue.put(out)  # Send the result
          except Exception as e:
              logger.error(f"Inference error: {e}")
              response_queue.put(e)  # Send the exception

def single_gpu_run(
    rank,
    task_type: TaskType,
    model_id: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    quant_model: bool = True,
    world_size: int = 1,
    is_offload: bool = True,
    offload_config: OffloadConfig = OffloadConfig(),
    enable_cfg_parallel: bool = True,
):
    """The target function for the child process."""
    pipe = SkyReelsVideoSingleGpuInfer(
        task_type=task_type,
        model_id=model_id,
        quant_model=quant_model,
        local_rank=rank,
        world_size=world_size,
        is_offload=is_offload,
        offload_config=offload_config,
        enable_cfg_parallel=enable_cfg_parallel,
    )
    pipe.inference(request_queue, response_queue)
class Predictor:  # Manages the child process
    def __init__(self):
        self.process = None
        self.request_queue = None
        self.response_queue = None
        self.config = None  # Store config here

    def initialize(self,
        task_type: TaskType,
        model_id: str,
        quant_model: bool = True,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        enable_cfg_parallel: bool = True,):

        self.config = {  # Store ALL config in a dict
            "task_type": task_type,
            "model_id": model_id,
            "quant_model": quant_model,
            "world_size": world_size,
            "is_offload": is_offload,
            "offload_config": offload_config,
            "enable_cfg_parallel": enable_cfg_parallel,
        }

        # DO NOT create process or queues here!

    def _ensure_initialized(self):
        """Creates and starts the process (if it doesn't exist)."""
        if self.process is None:
            if self.config is None:
                raise RuntimeError("Predictor not initialized. Call initialize() first.")

            with spawn_context() as ctx:
                self.request_queue = ctx.Queue()
                self.response_queue = ctx.Queue()
                self.process = ctx.Process(
                    target=single_gpu_run,
                    kwargs=self.config,  # Pass config as keyword arguments
                )
                self.process.start()
                logger.info(f"Started inference process with PID: {self.process.pid}")

                # Get ready signal, send init and warm up if needed.
                ready_msg = self.response_queue.get()
                logger.info(f"Process Ready msg: {ready_msg}")
                self.request_queue.put("INIT")
                init_response = self.response_queue.get()
                if isinstance(init_response,Exception):
                  raise init_response

                if self.config["offload_config"].compiler_transformer: #Warm up call
                  self.request_queue.put("WARMUP")
                  warmup_response = self.response_queue.get()
                  if isinstance(warmup_response, Exception):
                    raise warmup_response

    def infer(self, **kwargs):
        """Sends an inference request and returns the result."""
        self._ensure_initialized()  # Create process if needed

        self.request_queue.put(kwargs)
        result = self.response_queue.get()

        if isinstance(result, Exception):
            raise result
        return result

    def __del__(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            logger.info("Inference process terminated")
