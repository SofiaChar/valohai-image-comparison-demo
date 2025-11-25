
# 📸 Valohai Image Comparison Demo

This repository demonstrates how to use **Valohai’s Image Comparison** feature in a real YOLO-based computer vision workflow.

It includes:

- A **full pipeline**: preprocessing → training → prediction  
- A **prediction script with metadata** → enabling *grouped image comparison*  
- A **segmentation prediction script without metadata** → demonstrating *ungrouped* comparison  
- Examples of RGBA masks, overlays, and advanced visualization outputs

The goal is to show how different visualization outputs behave in Valohai’s UI, and how metadata enables much more powerful comparison experiences.

---

## 🧩 Repository Structure

```

├── data-preprocess.py      # Preprocess zip dataset → YOLO format
├── train.py                # YOLO training with Valohai tracking
├── evaluation.py           # Model evaluation (optional usage)
├── predict.py              # Prediction + metadata (GROUPED comparison)
├── predict-seg.py          # Prediction for segmentation (NO metadata)
├── helpers.py              # Small shared functions
├── valohai.yaml            # Pipeline + step definitions
├── valohai_watch.py        # Training logs → Valohai metadata
└── README.md
```

---

## 🏗️ 1. Pipeline Overview

The pipeline is defined in **valohai.yaml** and includes:

### **1. Data Preparation (`prepare_data`)**
- Downloads the Ships Aerial dataset  
- Unzips and restructures it  
- Produces YOLO-style `train/valid/test` splits  
- Writes a `data.yaml` describing paths  
- Allows custom dataset size per split

### **2. Training (`train_model`)**
- Runs YOLOv8 training using the preprocessed dataset  
- Streams training logs via `valohai_watch.py`  
- Produces model checkpoints, including `*best.pt`, prposes new version of the ship-detection model

### **3. Prediction (`prediction`)**
- Loads the chosen YOLO model  
- Runs inference on **test** and **validation** images  
- Produces three visualizations per image:
  - Original image  
  - Overlay (original + bounding boxes drawn by Ultralytics)  
  - RGBA bbox mask (transparent background, white opaque boxes)  
- Saves **Valohai image comparison metadata** to group all three outputs together

This step is where the *advanced grouping* logic happens.

---

## 🧠 2. Prediction with Metadata (Grouped Comparison)

The `predict.py` script demonstrates how to prepare outputs for Valohai’s **Advanced Image Comparison** interface.

For each image, it saves:

- `test_images/<name>.jpg`  
- `test_overlay_images/<name>.jpg`  
- `test_bbox_images/<name>.png` (RGBA: transparent background, white boxes)

Then, for each file, it writes a sidecar:

```
valohai.metadata.jsonl
```

Example:

```json
{
  "vhic_group": "test/IMG_123",
  "vhic_base": "test_images/IMG_123.jpg",
  "vhic_truth": "test_overlay_images/IMG_123.jpg",
  "vhic_name": "mask"
}
```

### ⭐ What this enables

In Valohai UI → **Images tab**:

- All outputs belonging to one source image (original, mask, overlay) appear:
  - In **one stack**  
  - With the base and comparison layers correctly preselected  
  - Allowing alpha blending, side-by-side comparison, and layer switching  
- You can select *Base + 1 comparison stack* to see perfectly grouped layers

This shows the *ideal workflow* for debugging model predictions visually.

---

## 🎨 3. Segmentation Prediction Without Metadata

The repository also includes:

```
predict-seg.py
```

This script:

- Runs inference with a segmentation-capable YOLO model  
- Saves:
  - segmentation masks  
  - bbox masks  
  - original images  
  - overlays  
- **Does not set the metadata for the image files to define the groups**

### Why?

This script demonstrates how **outputs behave without metadata**:

- Valohai will show each file as its own **separate stack**  
- No grouping  
- You can still compare them manually, but layer organization is simpler

This provides a clear contrast to the richer, grouped output of `predict.py`.

---

## 🖼️ 4. Understanding the Difference

| Scenario | Files Grouped? | UI Experience |
|---------|----------------|----------------|
| **predict.py** (with metadata) | ✅ Yes (original + overlay + RGBA mask) | One stack per image, toggleable layers, alpha blending |
| **predict-seg.py** (no metadata) | ❌ No | Each output is its own stack; manual pairing |

