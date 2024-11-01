import tkinter as tk
from tkinter import PhotoImage, Text
import subprocess


class App:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1080x1080")
        self.root.title("Image and Terminal")

        # Load and display image
        self.image_frame = tk.Frame(root, width=1080, height=1080)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.image = PhotoImage(
            file="./examples/screen/panel.png"
        )  # Replace with your image path
        self.image_label = tk.Label(self.image_frame, image=self.image)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Disable window decorations
        self.root.overrideredirect(True)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
