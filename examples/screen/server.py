from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from multiprocessing import Queue, get_context

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Create the prompt queue
ctx = get_context("spawn")
prompt_queue = ctx.Queue()
hsv_queue = ctx.Queue()
strength_queue = ctx.Queue()


# Flask route to serve the HTML page
@app.route("/")
def index():
    return render_template("index.html")


# SocketIO event for receiving prompt updates
@socketio.on("update_prompt")
def handle_update_prompt(data):
    prompt = data.get("prompt", "")
    colorName = data.get("colorName", "")
    color = data.get("colorCode", {"r": 0, "g": 0, "b": 0})
    r, g, b = color.values()

    prompt_queue.put([prompt, colorName])
    hsv_queue.put([int(r), int(g), int(b)])


# @socketio.on("update_hsv")
# def handle_update_csv(data):
#     hue = data.get("hue", 0)
#     sat = data.get("sat", 100)
#     val = data.get("val", 100)
#     # print(hue, sat, val)
#     hsv_queue.put([int(hue), int(sat), int(val)])
#     print(f"Received new prompt: {hue, sat, val}")


@socketio.on("update_rgb")
def handle_update_rgb(data):
    r = data.get("r", 0)
    g = data.get("g", 0)
    b = data.get("b", 0)
    # print(hue, sat, val)
    hsv_queue.put([int(r), int(g), int(b)])


@socketio.on("update_strength")
def handle_update_rgb(data):
    strength = data.get("strength", 1.6)

    strength_queue.put(
        int(strength * 10)
    )  # int가 아니면 queue가 동작 안할 수도 있어서 정수화 후 보냄


# Start the Flask app with SocketIO
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
