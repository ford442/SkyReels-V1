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
    def _load_model(
        self,
        model_id: str,
        base_model_id: str = "hunyuanvideo-community/HunyuanVideo",
        quant_model: bool = True,
    ):
        # No gpu_device parameter here
        logger.info(f"load model model_id:{model_id} quan_model:{quant_model}")
        text_encoder = LlamaModel.from_pretrained(
            base_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to("cpu")
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device="cpu",
        ).to("cpu")

        if quant_model:
            quantize_(text_encoder, float8_weight_only())  # No device here
            text_encoder.to("cpu")
            torch.cuda.empty_cache()  # Clear cache *after* moving to CPU
            quantize_(transformer, float8_weight_only())  # No device here
            transformer.to("cpu")
            torch.cuda.empty_cache()

        pipe = SkyreelsVideoPipeline.from_pretrained(
            base_model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
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
        self.task_type = task_type
        self.gpu_rank = local_rank
        self.model_id = model_id  # Store model_id for later
        self.quant_model = quant_model
        self.world_size = world_size
        self.is_offload = is_offload
        self.offload_config = offload_config
        self.enable_cfg_parallel = enable_cfg_parallel

        # Don't load the model here!
        self.pipe = None  # Initialize to None
        self.is_initialized = False

    def initialize(self):
        """Initializes the model and moves it to the GPU (if available)."""
        if self.is_initialized:
            return  # Already initialized

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot initialize model.")

        gpu_device = "cuda:0"  # Now we can safely assume CUDA is available
        self.pipe = self._load_model(
            model_id=self.model_id, quant_model=self.quant_model
        )

        max_batch_dim_size = 2 if self.enable_cfg_parallel and self.world_size > 1 else 1
        # Removed: max_ulysses_dim_size as it's not used
        logger.info(f"max_batch_dim_size: {max_batch_dim_size}")

        if self.is_offload:
          pass #Skip for now
            # Offload.offload(
            #     pipeline=self.pipe,
            #     config=self.offload_config,
            # )
        else:
            self.pipe.to(gpu_device)

        if self.offload_config.compiler_transformer:
            torch._dynamo.config.suppress_errors = True
            os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{self.offload_config.compiler_cache}_{self.world_size}"
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="max-autotune-no-cudagraphs",
                dynamic=True,
            )
            self.warm_up()
        self.is_initialized = True


    def warm_up(self):
        init_kwargs = {
            "prompt": "A woman is dancing in a room",
            "height": 544,
            "width": 960,
            "guidance_scale": 6,
            "num_inference_steps": 1,
            "negative_prompt": "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
            "num_frames": 97,
            "generator": torch.Generator("cuda").manual_seed(42),
            "embedded_guidance_scale": 1.0,
        }
        if self.task_type == TaskType.I2V:
            init_kwargs["image"] = Image.new("RGB", (544, 960), color="black")
        self.pipe(**init_kwargs)

    def damon_inference(self, request_queue: mp.Queue, response_queue: mp.Queue):
        # Initialize here, *after* we know the GPU is available
        self.initialize()
        response_queue.put(f"rank:{self.gpu_rank} ready")
        logger.info(f"rank:{self.gpu_rank} finish init pipe")
        while True:
            logger.info(f"rank:{self.gpu_rank} waiting for request")
            kwargs = request_queue.get()
            logger.info(f"rank:{self.gpu_rank} kwargs: {kwargs}")
            if "seed" in kwargs:
                kwargs["generator"] = torch.Generator("cuda").manual_seed(kwargs["seed"])
                del kwargs["seed"]
            start_time = time.time()
            assert (self.task_type == TaskType.I2V and "image" in kwargs) or self.task_type == TaskType.T2V
            out = self.pipe(**kwargs).frames[0]
            logger.info(f"rank:{self.gpu_rank} inference time: {time.time() - start_time}")
            response_queue.put(out) #Removed rank checking

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
    pipe.damon_inference(request_queue, response_queue)

class SkyReelsVideoInfer:
    def __init__(
        self,
        task_type: TaskType,
        model_id: str,
        quant_model: bool = True,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        enable_cfg_parallel: bool = True,
    ):
        self.world_size = world_size
        self.REQ_QUEUES = mp.Queue()  # Use normal queue.
        self.RESP_QUEUE = mp.Queue()
        assert self.world_size > 0, "gpu_num must be greater than 0"

        # Use a regular thread, but the key is the context manager in lauch_single_gpu_infer
        spawn_thread = threading.Thread(
            target=self.lauch_single_gpu_infer,
            args=(task_type, model_id, quant_model, world_size, is_offload, offload_config, enable_cfg_parallel),
            daemon=False,  # Still good practice to set this to False
        )
        spawn_thread.start()
        logger.info(f"Started multi-GPU thread with GPU_NUM: {world_size}")
        # Block and wait for the prediction process to start
        for _ in range(world_size):
            msg = self.RESP_QUEUE.get()
            logger.info(f"launch_multi_gpu get init msg: {msg}")

    def lauch_single_gpu_infer(
        self,
        task_type: TaskType,
        model_id: str,
        quant_model: bool = True,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        enable_cfg_parallel: bool = True,
    ):
        # Use the context manager here!
        with spawn_context() as ctx:
            print(f"Daemon status before spawn: {mp.current_process().daemon}")  # Debug print
            ctx.spawn(  # Use ctx.spawn, *not* mp.spawn
                single_gpu_run,
                nprocs=world_size,
                join=True,
                daemon=False,  # Controls child processes
                args=(
                    task_type,
                    model_id,
                    self.REQ_QUEUES,
                    self.RESP_QUEUE,
                    quant_model,
                    world_size,
                    is_offload,
                    offload_config,
                    enable_cfg_parallel,
                ),
            )
        logger.info(f"finish lanch multi gpu infer, world_size:{world_size}")

predictor = None

import spaces
@spaces.GPU(duration=90)
def init_predictor():
    global predictor
    if predictor is None: # Prevent multiple initializations.
      predictor = SkyReelsVideoInfer(
          task_type=TaskType.I2V,
          model_id="Skywork/SkyReels-V1-Hunyuan-I2V",
          quant_model=True,
          world_size=1,
          is_offload=True,
          offload_config=OffloadConfig(
              high_cpu_memory=True,
              parameters_level=True,
              compiler_transformer=False,
          ),
      )
    else:
      print("Predictor already initialized")
