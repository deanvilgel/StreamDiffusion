import os
import sys
import time
import threading
import math
from multiprocessing import Process, Queue, Pipe, get_context
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional
import torch
import PIL.Image
from streamdiffusion.image_utils import pil2tensor
import mss
import fire
import tkinter as tk
import server
from torchvision.transforms import functional as torchFunc
from torchvision.transforms import GaussianBlur as torchBlur
import warnings
from screeninfo import get_monitors

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper


inputs = []
top = 0
left = 0
import numpy as np


def printProgress(iteration, total, prefix="", suffix="", decimals=1, barLength=20):
    formatStr = "{0:." + str(decimals) + "f}"

    percent = formatStr.format(min(100 * (iteration / float(total)), 100))

    filledLength = int(round(barLength * iteration / float(total)))
    bar = "#" * filledLength + "-" * (barLength - filledLength)

    # Use carriage return (\r) to overwrite the same line
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")

    # Ensure the progress bar is updated for the last iteration
    # if iteration == total:
    #     sys.stdout.write("\n")
    sys.stdout.flush()


def start_flask_server(prompt_queue, hsv_queue, strength_queue):
    """
    Starts the Flask server defined in server.py.

    Parameters:
    prompt_queue (Queue): The queue to pass to the Flask server for prompt updates.
    """
    server.prompt_queue = prompt_queue  # Share the queue with the server
    server.hsv_queue = hsv_queue
    server.strength_queue = strength_queue
    server.socketio.run(server.app, host="0.0.0.0", port=5000)


def screen(
    event: threading.Event,
    width: int = 720,
    height: int = 405,
    monitor: Dict[str, int] = {"top": 0, "left": 0, "width": 720, "height": 405},
):
    global inputs
    # mss 0은 전체 모니터, 나머지 순서는 기준 모름 tk 는 디스플레이에 적힌대로 따라가는듯
    with mss.mss() as sct:
        while True:
            if event.is_set():
                print("terminate read thread")
                break
            img = sct.grab(
                sct.monitors[3]
            )  # 여기 숫자 변경하면 사용모니터 변경가능, 1 ,2, 3 중에 때려맞추셈
            img = PIL.Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            img.resize((height, width))
            inputs.append(pil2tensor(img))
    print("exit : screen")


def dummy_screen(
    width: int,
    height: int,
):
    root = tk.Tk()
    root.title("Press Enter to start")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)
    root.attributes("-alpha", 0.8)
    root.configure(bg="black")

    def destroy(event):
        root.destroy()

    root.bind("<Return>", destroy)

    def update_geometry(event):
        global top, left
        top = root.winfo_y()
        left = root.winfo_x()

    root.bind("<Configure>", update_geometry)
    root.mainloop()
    mornitor_used = get_monitors()[0]  # get mornitor input from 3rd mornitor
    return {
        "top": mornitor_used.x,
        "left": mornitor_used.y,
        "width": mornitor_used.width,
        "height": mornitor_used.height,
    }


def monitor_setting_process(
    width: int,
    height: int,
    monitor_sender: Connection,
) -> None:
    monitor = dummy_screen(width, height)
    monitor_sender.send(monitor)


def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    monitor_receiver: Connection,
    prompt_queue: Queue,
    hsv_queue: Queue,
    strength_queue: Queue,
    strength: float,
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """

    common_prompt = "'single color', 'realism',  'extremely realistic', 'simple background', 'center focused', 'vignette', 'high resolution', 'single object'"

    old_strength = strength = new_strength = 0.6

    current_prompt = prompt
    # target_hue = hue = 0.5
    # target_sat = sat = 1.0
    # target_val = val = 1.0
    target_r = r = 0.5
    target_g = g = 1.0
    target_b = b = 1.0

    prompt_lerp_threshold = 0.9
    prompt_lerp = 1
    hsv_lerp_threshold = 0.9

    strength_lerp = 1
    strength_lerp_threshold = 0.9
    is_print = True

    last_valid_image = None
    global inputs
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    stream.prepare(
        prompt=current_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
        strength=strength + 1,
    )

    monitor = monitor_receiver.recv()

    event = threading.Event()
    input_screen = threading.Thread(target=screen, args=(event, height, width, monitor))
    input_screen.start()
    time.sleep(5)

    while True:
        try:

            if not close_queue.empty():  # closing check
                break
            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue

            if not prompt_queue.empty():
                new_prompt_item = (
                    prompt_queue.get_nowait()
                )  # Non-blocking queue retrieval
                new_prompt = new_prompt_item[0]
                new_color = new_prompt_item[1]

            if not strength_queue.empty():
                new_strength = int(strength_queue.get_nowait()) / 10
                new_prompt = "" if round(new_strength, 2) == 0.0 else new_prompt

                print(f"\n\nNew Prompt : {new_prompt}")
                print(f"New Tint Color : {new_color}")
                print(f"Creativity : {round(new_strength,2)}")

            if new_prompt != current_prompt:
                current_prompt = new_prompt
                prompt_lerp = 1
            if strength != new_strength:
                old_strength = strength
                strength = new_strength
                strength_lerp = 1

            if not hsv_queue.empty():
                new_hsv = hsv_queue.get_nowait()

                target_r, target_g, target_b = new_hsv

                # new_hsv = [(x / 100) for x in new_hsv]
                # target_hue, target_sat, target_val = new_hsv
                # target_hue -= 0.5
                # print(f"hsv updated to: {target_hue}, {target_sat}, {target_val}")

            # hue = (
            #     hue * hsv_lerp_threshold + (1 - hsv_lerp_threshold) * target_hue
            #     if target_hue
            #     else hue
            # )
            # sat = (
            #     sat * hsv_lerp_threshold + (1 - hsv_lerp_threshold) * target_sat
            #     if target_sat
            #     else sat
            # )
            # val = (
            #     val * hsv_lerp_threshold + (1 - hsv_lerp_threshold) * target_val
            #     if target_val
            #     else val
            # )
            r = (
                r * hsv_lerp_threshold + (1 - hsv_lerp_threshold) * target_r
                if target_r
                else r
            )
            g = (
                g * hsv_lerp_threshold + (1 - hsv_lerp_threshold) * target_g
                if target_g
                else g
            )
            b = (
                b * hsv_lerp_threshold + (1 - hsv_lerp_threshold) * target_b
                if target_b
                else b
            )

            out_prompt = current_prompt

            start_time = time.time()
            sampled_inputs = []

            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])

            input_batch = torch.cat(sampled_inputs)

            input_batch = (input_batch + 1) / 2
            input_batch = torchFunc.resize(input_batch, (width, height))

            # overlay
            overlay_color = torch.tensor(
                [r / 255, g / 255, b / 255]
            )  # 순수한 붉은색 (R, G, B)
            overlay_strength = (
                strength_lerp if round(new_strength, 2) == 0.0 else 1 - strength_lerp
            )  # Set the strength between 0 (no overlay) and 1 (full overlay)

            overlay_batch = overlay_color.view(3, 1, 1).expand_as(input_batch)
            input_batch = (1 - overlay_strength) * input_batch + overlay_strength * (
                (1 - 2 * overlay_batch) * input_batch**2
                + 2 * overlay_batch * input_batch
            )

            # hsv adjustment
            # with torch.no_grad():
            #     input_batch = torchFunc.adjust_hue(input_batch, hue)
            #     input_batch = torchFunc.adjust_saturation(input_batch, sat)
            #     input_batch = torchFunc.adjust_brightness(input_batch, val)

            input_batch = torchFunc.autocontrast(input_batch)

            # noise_batch = torch.randn_like(input_batch)
            # input_batch = input_batch + (0.1**0.5) * noise_batch
            # input_batch = torchFunc.gaussian_blur(input_batch, 9)
            stream.stream.update_prompt(
                prompt="(" + out_prompt + ")," + common_prompt, alpha=prompt_lerp
            )

            lerp = strength_lerp * old_strength + (1 - strength_lerp) * strength + 1
            stream.stream.update_strength(strength=lerp)

            prompt_lerp = prompt_lerp * prompt_lerp_threshold
            strength_lerp = strength_lerp * strength_lerp_threshold

            if is_print:
                printProgress(1 - strength_lerp, 0.8, "Application Strength : ", "")

            if 1 - strength_lerp > 0.8:
                is_print = False
            else:
                is_print = True

            input_batch = (input_batch) * 2 - 1

            inputs.clear()
            output_images = stream.stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu()
            if frame_buffer_size == 1:
                output_images = [output_images]
            for output_image in output_images:

                queue.put_nowait(output_image)
                last_valid_image = output_image

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            break
        except Exception as e:
            # On any exception, use the last valid image instead of a blank screen
            if last_valid_image is not None:
                queue.put(last_valid_image, block=False)
            print(f"Error occurred: {e}. Using last valid image to avoid blank screen.")

    print("closing image_generation_process...")
    event.set()  # stop capture thread
    input_screen.join()
    print(f"fps: {fps}")


def main(
    ###### SF Background
    # model_id_or_path: str = "philz1337x/cyberrealistic-classic",
    ###### Anime Background
    model_id_or_path: str = "Lykon/DreamShaper",
    ###### Architecture
    # model_id_or_path: str = "a34384300/XSarchitectural-InteriorDesign-ForXSLora",
    lora_dict: Optional[Dict[str, float]] = {
        # "C:/Users/ERSATZ-CAMPUSTOWN/StreamDiffusion/models/LoRA/Unreal.safetensors": 1,
        "C:/Users/ERSATZ-CAMPUSTOWN/StreamDiffusion/models/LoRA/more_details.safetensors": 1,
        # "C:/Users/ERSATZ-CAMPUSTOWN/StreamDiffusion/models/LoRA/aidma-Image Upgrader-SD1.5-V0.1.safetensors": 1,
        # "C:/Users/ERSATZ-CAMPUSTOWN/StreamDiffusion/models/LoRA/Movie_Style_Backgrounds.safetensors": 1,
        # "C:/Users/ERSATZ-CAMPUSTOWN/StreamDiffusion/models/LoRA/XSarchitectural-38InteriorForBedroom.safetensors": 1,
        # "C:/Users/ERSATZ-CAMPUSTOWN/StreamDiffusion/models/LoRA/xsarchitectural-7.safetensors": 1,
    },
    prompt: str = "architectural rendering, beautiful interior",
    negative_prompt: str = "ugly, complex, dull, low quality, bad quality, blurry, low resolution, noise, fog, human face",
    frame_buffer_size: int = 1,
    width: int = 720,
    height: int = 405,
    acceleration: Literal["none", "xformers", "tensorrt"] = "xformers",
    use_denoising_batch: bool = True,
    seed: int = 2,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    guidance_scale: float = 1.4,
    delta: float = 0.5,
    do_add_noise: bool = True,
    enable_similar_image_filter: bool = False,
    similar_image_filter_threshold: float = 0.95,
    similar_image_filter_max_skip_frame: float = 10,
    strength: float = 1.6,
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    ctx = get_context("spawn")
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    prompt_queue = ctx.Queue()  # Queue for updating prompts in real time
    hsv_queue = ctx.Queue()
    strength_queue = ctx.Queue()

    close_queue = Queue()

    monitor_sender, monitor_receiver = ctx.Pipe()
    flask_process = Process(
        target=start_flask_server,
        args=(
            prompt_queue,
            hsv_queue,
            strength_queue,
        ),
    )
    flask_process.start()

    process1 = ctx.Process(
        target=image_generation_process,
        args=(
            queue,
            fps_queue,
            close_queue,
            model_id_or_path,
            lora_dict,
            prompt,
            negative_prompt,
            frame_buffer_size,
            width,
            height,
            acceleration,
            use_denoising_batch,
            seed,
            cfg_type,
            guidance_scale,
            delta,
            do_add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            monitor_receiver,
            prompt_queue,
            hsv_queue,
            strength_queue,
            strength,
        ),
    )
    process1.start()

    moniter_width = 1920
    moniter_height = 1080
    monitor_process = ctx.Process(
        target=monitor_setting_process,
        args=(
            moniter_width,
            moniter_height,
            monitor_sender,
        ),
    )
    monitor_process.start()
    monitor_process.join()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    # terminate
    process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    process1.join(5)  # with timeout
    if process1.is_alive():
        print("process1 still alive. force killing...")
        process1.terminate()  # force kill...
        flask_process.terminate()
    process1.join()
    print("process1 terminated.")


if __name__ == "__main__":
    fire.Fire(main)
