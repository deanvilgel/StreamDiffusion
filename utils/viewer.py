import os
import sys
import threading
import time
import tkinter as tk
from screeninfo import get_monitors
from multiprocessing import Queue
from typing import List
from PIL import Image, ImageTk
from streamdiffusion.image_utils import postprocess_image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

screen_order = 1


def update_image(image_data: Image.Image, label: tk.Label) -> None:
    """
    Update the image displayed on a Tkinter label.

    Parameters
    ----------
    image_data : Image.Image
        The image to be displayed.
    label : tk.Label
        The labels where the image will be updated.
    """
    width = 1920
    height = 1080
    image_data = image_data.resize(size=(width, height))
    tk_image = ImageTk.PhotoImage(image_data, size=width)
    label.configure(image=tk_image, width=width, height=height)
    label.image = tk_image  # keep a reference


def _receive_images(
    queue: Queue, fps_queue: Queue, label: tk.Label, fps_label: tk.Label
) -> None:
    """
    Continuously receive images from a queue and update the labels.

    Parameters
    ----------
    queue : Queue
        The queue to receive images from.
    fps_queue : Queue
        The queue to put the calculated fps.
    label : tk.Label
        The label to update with images.
    fps_label : tk.Label
        The label to show fps.
    """
    while True:
        try:
            if not queue.empty():
                label.after(
                    0,
                    update_image,
                    postprocess_image(queue.get(block=False), output_type="pil")[0],
                    label,
                )
            if not fps_queue.empty():
                fps_label.config(text=f"FPS: {fps_queue.get(block=False):.2f}")

            time.sleep(0.0005)
        except KeyboardInterrupt:
            return


def receive_images(queue: Queue, fps_queue: Queue) -> None:
    """
    Setup the Tkinter window and start the thread to receive images.

    Parameters
    ----------
    queue : Queue
        The queue to receive images from.
    fps_queue : Queue
        The queue to put the calculated fps.
    """
    # Get monitor information
    monitors = get_monitors()

    # Check if a second monitor exists
    if len(monitors) > 1 and screen_order == 2:
        first_monitor = monitors[0]
        second_monitor = monitors[1]  # second monitor

        root = tk.Tk()
        # Set window geometry to match the second monitor's resolution and position
        root.geometry(
            f"{second_monitor.width}x{second_monitor.height}+{first_monitor.x}+{first_monitor.y}"
        )

        def toggle_fullscreen(event=None):
            # Toggle fullscreen while ensuring the window stays on the second monitor
            if not root.attributes("-fullscreen"):
                # Enter fullscreen mode
                root.geometry(
                    f"{second_monitor.width}x{second_monitor.height}+{first_monitor.x}+{first_monitor.y}"
                )
                root.attributes("-fullscreen", True)
            else:
                # Exit fullscreen mode
                root.attributes("-fullscreen", False)
                root.geometry(
                    f"{second_monitor.width}x{second_monitor.height}+{first_monitor.x}+{first_monitor.y}"
                )

    else:
        first_monitor = monitors[0]
        root = tk.Tk()
        # Set window geometry to match the second monitor's resolution and position
        root.geometry(f"{first_monitor.width}x{first_monitor.height}")

        def toggle_fullscreen(event=None):
            # Toggle fullscreen while ensuring the window stays on the second monitor
            if not root.attributes("-fullscreen"):
                # Enter fullscreen mode
                root.geometry(f"{first_monitor.width}x{first_monitor.height}")
                root.attributes("-fullscreen", True)
            else:
                # Exit fullscreen mode
                root.attributes("-fullscreen", False)
                root.geometry(f"{first_monitor.width}x{first_monitor.height}")

    # Bind the F11 key to toggle fullscreen
    root.bind("<F11>", toggle_fullscreen)
    root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

    root.title("Image Viewer")
    label = tk.Label(root)
    fps_label = tk.Label(root, text="FPS: 0")
    label.grid(column=0)
    fps_label.grid(column=1)

    def on_closing():
        print("window closed")
        root.quit()  # stop event loop
        return

    thread = threading.Thread(
        target=_receive_images,
        args=(queue, fps_queue, label, fps_label),
        daemon=True,
    )
    thread.start()

    try:
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except KeyboardInterrupt:
        return
