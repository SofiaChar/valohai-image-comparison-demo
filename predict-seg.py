import os
import cv2
from pathlib import Path
from io import BytesIO

import numpy as np
import requests
from PIL import Image, ImageDraw
from ultralytics import YOLO
import valohai

# ---------------- CONFIG ----------------

MODEL_PATH = valohai.parameters('yolo_model_name').value
OUTPUT_DIR = "/valohai/outputs/"
OUTPUT_ROOT = "/valohai/outputs/"

IMAGE_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
]

CONF_THRES = 0.25


#
# ---------------- HELPERS ----------------

def download_image(url: str) -> Image.Image:
    print(f"Downloading: {url}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img


def save_image(img: Image.Image, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def create_blank_rgba(size):
    """Return a fully-transparent RGBA canvas."""
    w, h = size
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


# ---------------- MAIN LOGIC ----------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    for idx, url in enumerate(IMAGE_URLS, start=1):
        try:
            img = download_image(url)
        except Exception as e:
            print(f"[{idx}] Failed to download {url}: {e}")
            continue

        base_name = f"img_{idx}"
        w, h = img.size

        # ---------- ORIGINAL ----------
        orig_path = os.path.join(OUTPUT_DIR, f"{base_name}_orig.png")
        save_image(img, orig_path)
        print(f"[{idx}] Saved original -> {orig_path}")

        # ---------- RUN SEGMENTATION ----------
        print(f"[{idx}] Running model...")
        result = model(np.array(img), conf=CONF_THRES)[0]

        # ---------- 1) BOXES-ONLY IMAGE (RGBA) ----------
        boxes_canvas = create_blank_rgba(img.size)
        draw = ImageDraw.Draw(boxes_canvas)

        num_boxes = 0
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                # Clamp to valid range
                x1 = float(max(0, min(x1, w - 1)))
                x2 = float(max(0, min(x2, w - 1)))
                y1 = float(max(0, min(y1, h - 1)))
                y2 = float(max(0, min(y2, h - 1)))

                # Green rectangles with full alpha
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=(0, 255, 0, 255),  # RGBA
                    width=4,
                )
                num_boxes += 1

        print(f"[{idx}] Boxes detected: {num_boxes}")
        box_path = os.path.join(OUTPUT_DIR, f"{base_name}_boxes.png")
        save_image(boxes_canvas, box_path)
        print(f"[{idx}] Saved boxes (RGBA) -> {box_path}")

        # ---------- 2) SEGMENTATION MASK (RGBA) ----------
        # *** THIS BLOCK IS EXACTLY YOUR WORKING VERSION, UNCHANGED ***
        mask_canvas = np.zeros((h, w, 4), dtype=np.uint8)  # transparent

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()   # (N, Hm, Wm)
            num_masks = masks.shape[0]

            colors = [
                (255, 0, 255),   # magenta
                (0, 255, 255),   # cyan
                (255, 255, 0),   # yellow
                (0, 255, 0),     # green
                (255, 128, 0),   # orange
            ]

            for i, m in enumerate(masks):
                color = colors[i % len(colors)]

                # Resize mask to original image size
                m_img = Image.fromarray((m * 255).astype(np.uint8), mode="L")
                m_img = m_img.resize((w, h), resample=Image.NEAREST)
                binary = np.array(m_img) > 127  # boolean mask (H, W)

                mask_canvas[binary] = [color[0], color[1], color[2], 180]

            print(f"[{idx}] Segmentation masks applied: {num_masks}")
        else:
            print(f"[{idx}] No masks found.")

        mask_pil = Image.fromarray(mask_canvas, mode="RGBA")
        seg_path = os.path.join(OUTPUT_DIR, f"{base_name}_segmentation.png")
        save_image(mask_pil, seg_path)
        print(f"[{idx}] Saved segmentation masks (RGBA) -> {seg_path}")

    print("\nDone! For each image you now have:")
    print("- *_orig.png")
    print("- *_boxes.png         (transparent BG + green boxes)")
    print("- *_segmentation.png  (transparent BG + colored masks)")
    print("Ready for Valohai image comparison ✨")


if __name__ == "__main__":
    main()