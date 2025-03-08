import gradio as gr
import argparse
import sys
import time
import os
import random
sys.path.append(".")
from skyreelsinfer import TaskType
from skyreelsinfer.offload import OffloadConfig
from skyreelsinfer.skyreels_video_infer import SkyReelsVideoInfer
from diffusers.utils import export_to_video
from diffusers.utils import load_image

predictor = None
task_type = None

def get_transformer_model_id(task_type:str) -> str:
    return "Skywork/SkyReels-V1-Hunyuan-I2V" if task_type == "i2v" else "Skywork/SkyReels-V1-Hunyuan-T2V"

def init_predictor(task_type:str, gpu_num:int=1):
    global predictor
    predictor = SkyReelsVideoInfer(
        task_type= TaskType.I2V if task_type == "i2v" else TaskType.T2V,
        model_id=get_transformer_model_id(task_type),
        quant_model=True,
        world_size=gpu_num,
        is_offload=True,
        offload_config=OffloadConfig(
            high_cpu_memory=True,
            parameters_level=True,
            compiler_transformer=False,
        )
    )

def generate_video(prompt, seed, image=None):
    global task_type
    print(f"image:{type(image)}")

    if seed == -1:
        random.seed(time.time())
        seed = int(random.randrange(4294967294))
    
    kwargs = {
        "prompt": prompt,
        "height": 512,
        "width": 512,
        "num_frames": 97,
        "num_inference_steps": 30,
        "seed": seed,
        "guidance_scale": 6.0,
        "embedded_guidance_scale": 1.0,
        "negative_prompt": "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
        "cfg_for": False,
    }

    if task_type == "i2v":
        assert image is not None, "please input image"
        kwargs["image"] = load_image(image=image)
    global predictor
    output = predictor.inference(kwargs)
    save_dir = f"./result/{task_type}"
    os.makedirs(save_dir, exist_ok=True)
    video_out_file = f"{save_dir}/{prompt[:100].replace('/','')}_{seed}.mp4"
    print(f"generate video, local path: {video_out_file}")
    export_to_video(output, video_out_file, fps=24)
    return video_out_file, kwargs


def create_gradio_interface(task_type):
    """Create a Gradio interface based on the task type."""
    if task_type == "i2v":
        with gr.Blocks() as demo:
            with gr.Row():
                image = gr.Image(label="Upload Image", type="filepath")
                prompt = gr.Textbox(label="Input Prompt")
                seed = gr.Number(label="Random Seed", value=-1)
            submit_button = gr.Button("Generate Video")
            output_video = gr.Video(label="Generated Video")
            output_params = gr.Textbox(label="Output Parameters")

            # Submit button logic
            submit_button.click(
                fn=generate_video,
                inputs=[prompt, seed, image],
                outputs=[output_video, output_params],
            )

    elif task_type == "t2v":
        with gr.Blocks() as demo:
            with gr.Row():
                prompt = gr.Textbox(label="Input Prompt")
                seed = gr.Number(label="Random Seed", value=-1)
            submit_button = gr.Button("Generate Video")
            output_video = gr.Video(label="Generated Video")
            output_params = gr.Textbox(label="Output Parameters")

            # Submit button logic
            submit_button.click(
                fn=generate_video,
                inputs=[prompt, seed],
                outputs=[output_video, output_params],  # Pass task_type as additional input
            )

    return demo

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Gradio app with a specific task type.")
    parser.add_argument("--task_type", type=str, choices=["i2v", "t2v"], required=True, help="Task type: i2v or t2v")
    parser.add_argument("--gpu_num", type=int, default=1)
    args = parser.parse_args()
    # Create and launch Gradio interface
    task_type = args.task_type
    init_predictor(task_type=task_type, gpu_num=args.gpu_num)
    demo = create_gradio_interface(args.task_type)
    demo.launch(share=True)
