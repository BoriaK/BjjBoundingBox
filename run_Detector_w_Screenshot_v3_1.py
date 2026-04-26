import atexit
import sys
import os
import torch
from Common_Files.Models import Detector
import cv2
import numpy as np
import torchvision.transforms.functional as Ftv
from PIL import Image, ImageOps
import mss
from pynput import keyboard
import win32clipboard
import win32con
import io
from pynput import keyboard as kb
import shutil
import tkinter as tk
import threading

# This script is testing the detector, to detect Furniture and place bounding box around them

'''adding types to arguments'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_classes = 2
is_train = False
model = Detector(n_classes, device)
if device == "cpu":
    checkpoint = torch.load('./checkpoints/bjj_19' + '.pth')
else:
    checkpoint = torch.load('./checkpoints/bjj_19' + '.pth')
model.load_state_dict(checkpoint['model_state'])
model.eval()

# --- Exit notification ---
_exit_info = {"code": 0, "reason": ""}


def _show_exit_notification(message, duration=5):
    """Blocking yellow toast shown when the script exits (runs in atexit)."""
    try:
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes('-topmost', True)

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        window_width = 500
        window_height = 70
        x = screen_width - window_width - 20
        y = screen_height - window_height - 50

        root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        frame = tk.Frame(root, bg="#FFD700", borderwidth=2, relief="raised")
        frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        label = tk.Label(
            frame,
            text=message,
            font=("Arial", 11, "bold"),
            bg="#FFD700",
            fg="#333333",
            wraplength=window_width - 20,
        )
        label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        root.after(duration * 1000, root.destroy)
        root.mainloop()
    except Exception:
        pass


def _on_exit():
    code = _exit_info["code"]
    reason = _exit_info["reason"]
    msg = f"Process finished with exit code {code}"
    if reason:
        msg += f"  —  {reason}"
    _show_exit_notification(msg)


atexit.register(_on_exit)
# -------------------------


# ts = time.time()
# root = r'./Dataset/Sub_Set_v01'

# Get all monitors information
def get_screen_info():
    with mss.mss() as sct:
        monitors = sct.monitors
        # for i, monitor in enumerate(monitors):
        #     print(
        #         f"Monitor: {i}, Left: {monitor['left']}, Top: {monitor['top']}, Width: {monitor['width']}, Height: {monitor['height']}")
        return monitors


def take_screenshot(monitor_index=1):
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor index {monitor_index} is out of range. Available monitors: {len(monitors) - 1}")

        monitor = monitors[monitor_index]
        # a correction to the partial screenshot on the small monitor
        if len(monitors) > 2 and monitor_index == 1:
            monitor['width'] = monitor['width'] + 380
            monitor['height'] = monitor['height'] + 220

        # Take a screenshot
        screenshot = sct.grab(monitor)
        # Convert the screenshot to a PIL image
        screenshot_pil = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        # for debug:
        # screenshot_pil.show(title="Screenshot")
        ##################################
        # screenshot_np = np.array(screenshot)

    return screenshot_pil


def detect_objects(pil_image):
    # Convert the NumPy array back to a PIL Image for transformation
    # image_pil = Image.fromarray(image)

    # Apply the transformations to the image
    image_tensor = Ftv.to_tensor(pil_image)

    # Add a batch dimension
    # image_tensor = image_tensor.unsqueeze(0)

    # Perform the object detection
    with torch.no_grad():
        predictions = model([image_tensor.to(device)], [], is_train)

    return predictions


def process_predictions(predictions):
    boxes = []
    labels = []
    scores = []

    # Get the scores and find the index of the highest score
    scores_list = predictions[0]['scores'].tolist()
    if scores_list:
        max_score_index = scores_list.index(max(scores_list))

        # Append only the highest scoring prediction
        boxes.append(predictions[0]['boxes'][max_score_index].tolist())
        labels.append(predictions[0]['labels'][max_score_index].item())
        scores.append(predictions[0]['scores'][max_score_index].item())

    return boxes, labels, scores


def draw_boxes(pil_image, boxes):
    for b in boxes:
        start_point = (int(np.round(b[0])), int(np.round(b[1])))
        end_point = (int(np.round(b[2])), int(np.round(b[3])))
        color = (255, 0, 0)  # Red color for the rectangle
        thickness = 2
        im_np_array = np.array(pil_image)
        # Convert BGR back to RGB
        im_np_array = cv2.cvtColor(im_np_array, cv2.COLOR_BGR2RGB)
        im = cv2.rectangle(im_np_array, start_point, end_point, color, thickness)

    return im


def crop_and_save_image(pil_image, box, filename="cropped_image.png"):
    # Crop the image using the bounding box
    cropped_image = pil_image.crop((box[0], box[1], box[2], box[3]))

    # Convert cm to pixels (assuming 96 DPI)
    dpi = 96  # standard screen DPI
    target_cm_height = 4  # resize height to 4 cm
    target_height_pixels = int((target_cm_height / 2.54) * dpi)  # Convert 4 cm to pixels

    # Get the aspect ratio of the cropped image
    aspect_ratio = cropped_image.width / cropped_image.height

    # Calculate the new width while maintaining the aspect ratio
    target_width_pixels = int(target_height_pixels * aspect_ratio)

    # Resize the image to the new width and height (keeping proportional width)
    cropped_image_resized = cropped_image.resize((target_width_pixels, target_height_pixels))

    # Ensure the folder exists, or create it
    if not os.path.exists('./Results_for_Detector/'):
        os.makedirs('./Results_for_Detector/')

    # Save the resized image
    cropped_image_resized.save('./Results_for_Detector/' + filename)
    print(f"Cropped and resized image saved as {filename} with height {target_cm_height} cm and proportional width")

    return cropped_image_resized


def copy_image_to_clipboard(cropped_pil_image):
    output = io.BytesIO()
    cropped_pil_image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()

    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32con.CF_DIB, data)
    win32clipboard.CloseClipboard()
    print("Cropped image copied to clipboard.")


def show_message(message, duration=5, bg_color="#4CAF50", fg_color="white", window_width=400, window_height=80):
    """Display a small message window that automatically closes after the specified duration."""
    def create_and_show():
        # Create a new Tkinter window
        root = tk.Tk()
        root.overrideredirect(True)  # Remove window decorations
        root.attributes('-topmost', True)  # Keep the window on top

        # Get screen width and height to position the window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Position window at the bottom right corner with some padding
        x_position = screen_width - window_width - 20  # 20 pixels from the right edge
        y_position = screen_height - window_height - 50  # 50 pixels from the bottom

        root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        # Create a frame with padding and border
        frame = tk.Frame(root, bg=bg_color, borderwidth=2, relief="raised")
        frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Add a label with the message
        label = tk.Label(
            frame,
            text=message,
            font=("Arial", 12, "bold"),
            bg=bg_color,
            fg=fg_color,
            wraplength=window_width - 20
        )
        label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Schedule the window to close after the specified duration
        root.after(duration * 1000, root.destroy)

        # Run the Tkinter event loop
        root.mainloop()

    # Run the message window in a separate thread to avoid blocking
    thread = threading.Thread(target=create_and_show, daemon=True)
    thread.start()


def on_press(key):
    try:
        if key == kb.Key.insert:
            # if key.char == 'u':

            monitor_index = 1  # Change this to the index of the screen you want to capture
            print(f"Taking screenshot of monitor {monitor_index}...")
            image = take_screenshot(monitor_index)
            print("Detecting objects...")
            predictions = detect_objects(image)
            boxes, labels, scores = process_predictions(predictions)
            if not boxes:
                print("No objects detected. Quitting...")
                _exit_info["reason"] = "No objects detected"
                return False  # Stop the listener and quit the script

            print(f"Detected {len(boxes)} objects:")
            for i in range(len(boxes)):
                print(f"Object {i + 1}:")
                print(f"  Bounding Box: {boxes[i]}")
                print(f"  Label: {labels[i]}")
                print(f"  Score: {scores[i]:.4f}")

            # Draw boxes on the image
            image_with_boxes = draw_boxes(image, boxes)

            # Display the image with bounding boxes using OpenCV
            # cv2.namedWindow('Screenshot with Bounding Boxes',
            #                 cv2.WINDOW_NORMAL)  # to make the image fit the display window when printing
            # cv2.imshow("Screenshot with Bounding Boxes", image_with_boxes)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # plt.show()

            # cv2.imwrite('./Results_for_Detector/' + 'from_screenShot.png', image_with_boxes)

            # Crop and save the image using the bounding box of the highest scored object
            cropped_image = crop_and_save_image(image, boxes[0])
            copy_image_to_clipboard(cropped_image)

            # Show success message
            show_message("✓ Screenshot saved and copied to clipboard!", duration=5)

        # if key.char == 'q':
        if key == kb.Key.esc:
            print("Quitting...")
            _exit_info["reason"] = "User pressed ESC"
            return False

    except AttributeError:
        pass


def delete_results_folder(folder_path='./Results_for_Detector/'):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def main():
    print("Press 'Insert' to take a screenshot and detect objects.")
    print("Press 'ESC' to quit.")

    # Show startup hint on screen
    show_message(
        "⌨  Press 'Insert' to detect objects.\nPress 'ESC' to quit.",
        duration=6,
        bg_color="#1565C0",
        fg_color="white",
        window_width=420,
        window_height=90,
    )

    # screens = get_screen_info()
    # print("Available monitors:")u
    # for i, screen in enumerate(screens):
    #     print(f"Monitor {i}: Resolution: {screen['width']}x{screen['height']}")

    try:
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except Exception as e:
        _exit_info["code"] = 1
        _exit_info["reason"] = str(e)

    # Delete the folder at the end of the program
    delete_results_folder()


if __name__ == "__main__":
    main()

