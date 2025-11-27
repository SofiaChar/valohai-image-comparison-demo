from ultralytics import YOLO
import valohai
import shutil
import os
import json

import cv2
import numpy as np


RUNS_VAL_DIR = "runs/detect/val"
METADATA_PATH = "/valohai/outputs/valohai.metadata.jsonl"
RUNS_VAL_DIR = "runs/detect/val"
OUTPUT_ROOT = "/valohai/outputs"  # used for rel paths in metadata


def save_val_plots_to_outputs():
    """
    Copy Ultralytics' built-in validation plots from runs/detect/val
    into a Valohai output folder called 'detections'.
    """
    if not os.path.isdir(RUNS_VAL_DIR):
        print(f"Directory {RUNS_VAL_DIR} not found, skipping copy.")
        return

    save_path = valohai.outputs("detections")
    print("Saving Ultralytics val plots to:", save_path)

    for f in os.listdir(RUNS_VAL_DIR):
        src = os.path.join(RUNS_VAL_DIR, f)
        if os.path.isfile(src):
            dst = save_path.path(f)
            shutil.copy(src, dst)


def write_sidecar(image_abs_path: str, metadata: dict):
    """
    Write sidecar metadata JSON next to an image:
    """
    sidecar_path = image_abs_path + ".metadata.json"
    os.makedirs(os.path.dirname(sidecar_path), exist_ok=True)
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)


def generate_custom_plots(model):
    """
    For each split ('test' and 'valid'), run prediction and:
      - save the original image
      - save bbox-only visualization on white background
      - save overlay visualization using Ultralytics' built-in plotting

    For all three, write sidecar metadata with vhic_* fields so Valohai
    image comparison UI can group and compare them.
    """

    # Adjust these if your folder structure differs
    split_sources = {
        "test": "/valohai/inputs/test/prep_dataset/test/images",
        "valid": "/valohai/inputs/valid/prep_dataset/valid/images",
    }

    for split_name, root in split_sources.items():
        if not os.path.isdir(root):
            fallback = f"/valohai/inputs/{split_name}"
            print(f"{root} not found for split '{split_name}', falling back to {fallback}")
            root = fallback

        print(f"Running custom predictions for split '{split_name}' on: {root}")

        orig_output = valohai.outputs(f"{split_name}_images")
        bbox_output = valohai.outputs(f"{split_name}_bbox_images")
        overlay_output = valohai.outputs(f"{split_name}_overlay_images")

        results = model.predict(
            source=root,
            save=False,
            stream=True,
            verbose=False,
        )

        for i, r in enumerate(results):
            orig = r.orig_img  # numpy array (H, W, 3), BGR
            img_path = r.path  # original image path as string
            img_name = os.path.basename(img_path)
            img_stem, _ = os.path.splitext(img_name)

            # --- Save original image copy ---
            orig_abs = orig_output.path(img_name)
            cv2.imwrite(orig_abs, orig)

            # --- Create blank canvas for bbox-only visualization ---
            h, w, _ = orig.shape
            canvas = np.zeros((h, w, 4), dtype=np.uint8)

            if r.boxes is None or len(r.boxes) == 0:
                print(f"[{split_name}] No boxes for {img_name}")
            else:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                for (x1, y1, x2, y2) in boxes_xyxy:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0, 255), 2, lineType=cv2.LINE_8)

            # --- Save bbox image as PNG ---
            bbox_png_name = f"{img_stem}.png"
            bbox_abs = bbox_output.path(bbox_png_name)
            cv2.imwrite(bbox_abs, canvas)

            # --- Overlay image using Ultralytics' own plotting ---
            overlay = r.plot()  # BGR numpy image with boxes drawn
            overlay_abs = overlay_output.path(img_name)
            cv2.imwrite(overlay_abs, overlay)

            # ---------- Image comparison metadata via sidecars ----------
            # Relative paths from /valohai/outputs, which is what Valohai uses
            rel_orig = os.path.relpath(orig_abs, OUTPUT_ROOT)
            rel_overlay = os.path.relpath(overlay_abs, OUTPUT_ROOT)

            # Group per split + image, so you see e.g. "test/GE_690_jpg.rf..."
            group_name = f"{split_name}/{img_stem}"

            # We'll use original as base and overlay as truth
            base_file = rel_orig
            truth_file = rel_overlay

            # Original
            write_sidecar(
                orig_abs,
                {
                    "vhic_group": group_name,
                    "vhic_base": base_file,
                    "vhic_truth": truth_file,
                    "vhic_name": f"{split_name} original",
                },
            )

            # Overlay (predictions on original)
            write_sidecar(
                overlay_abs,
                {
                    "vhic_group": group_name,
                    "vhic_base": base_file,
                    "vhic_truth": truth_file,
                    "vhic_name": f"{split_name} overlay",
                },
            )

            # Bbox-only visualization
            write_sidecar(
                bbox_abs,
                {
                    "vhic_group": group_name,
                    "vhic_base": base_file,
                    "vhic_truth": truth_file,
                    "vhic_name": f"{split_name} bbox-only",
                },
            )

        print(f"Custom plots for split '{split_name}' saved to:")
        print(" - originals:", orig_output)
        print(" - bbox-only:", bbox_output)
        print(" - overlays :", overlay_output)

def evaluate_yolo():
    # Load a model from Valohai input
    model_path = valohai.inputs("model").path()
    print("Loading model from:", model_path)
    model = YOLO(model_path)

    # === 1) Run validation on test split (metrics + built-in plots) ===
    metrics = model.val(
        data="/valohai/inputs/data_yaml/data.yaml",
        split="test",
        plots=True,
    )

    print("map50-95:", metrics.box.map)
    print("map50:", metrics.box.map50)
    print("map75:", metrics.box.map75)
    print("List map50-95 of each category:", metrics.box.maps)

    # Ultralytics validation plots (PR curves, confusion matrix, etc.) saved outputs
    save_val_plots_to_outputs()

    # Run custom prediction pass for both test + valid
    generate_custom_plots(model)


if __name__ == "__main__":
    evaluate_yolo()
