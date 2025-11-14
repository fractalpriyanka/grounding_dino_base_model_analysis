# Zero-Shot Object Detection using Grounding DINO

**One-line:** A promptable object detection pipeline using **Grounding DINO (GDINO)** for zero-shot detection of objects described via natural language.

---

## Table of Contents

1. Overview
2. Problem Statement
3. Objectives
4. Pipeline Architecture
5. Methodology
6. Dataset & Resources
7. Quick Start
8. Project Structure
9. Dependencies
10. Notes & Recommendations

---

## 1. Overview

Traditional object detection models like YOLO or Faster R-CNN are limited to a fixed set of predefined classes. Grounding DINO (GDINO) overcomes this limitation by combining vision and language transformers, enabling **open-set detection** where objects can be localized based on textual prompts. This project builds a zero-shot detection pipeline using GDINO.

---

## 2. Problem Statement

Use the Grounding DINO model to perform **prompt-based object detection** on a set of natural or domain-specific images. Given a natural language prompt (e.g., "dog", "bottle", "tumor"), the model should detect and localize relevant instances in the image with bounding boxes.

---

## 3. Objectives

- Implement a **zero-shot object detection pipeline** using Grounding DINO.
- Evaluate detection ability on **unseen or rare classes** using natural language descriptions.
- Explore the impact of **prompt formulation and variability** on detection results.
- Visualize and analyze **both correct detections and failure modes** for deeper insights.

---

## 4. Pipeline Architecture

```
+------------------+
|   Input Image    |
+---------+--------+
          |
          v
+---------+--------+
|  Text Prompt(s)  |
+---------+--------+
          |
          v
+---------+--------+
| Preprocessor     |
| (Image + Text)   |
+---------+--------+
          |
          v
+---------+--------+
| Grounding DINO   |
| (Vision + Lang)  |
+---------+--------+
          |
          v
+---------+--------+
| Postprocessing   |
| (Boxes + Scores) |
+---------+--------+
          |
   +------+------+
   |             |
   v             v
Visualization   Evaluation
(PIL/Matplotlib) (Qualitative/Quantitative)
```

**Steps:**

1. Provide an image and a natural language prompt.
2. GDINO processes both inputs and returns bounding boxes + confidence scores.
3. Postprocessing applies thresholding and formatting.
4. Results are visualized and evaluated (mAP if ground truth available).

---

## 5. Methodology

1. **Dataset Preparation:** Select COCO 2017 validation set, Open Images subset, or the custom Kaggle dataset. Ensure images are preprocessed to the model’s required format.
2. **Model Setup:** Clone [GroundingDINO repo](https://github.com/IDEA-Research/GroundingDINO) or load pretrained models (e.g., Swin-B) via Hugging Face.
3. **Prompt-based Inference:** Provide prompts like `"cat"`, `"laptop"`, or domain-specific terms (e.g., `"surgical mask"`, `"tumor"`).
4. **Evaluation:**

   - **Qualitative:** Overlay bounding boxes on test images.
   - **Quantitative:** Compute mAP if annotations exist.

5. **Prompt Variability:** Experiment with synonyms, short vs. detailed descriptions to test sensitivity.

---

## 6. Dataset & Resources

- **Dataset (Kaggle):** [Zero-Shot Dataset](https://www.kaggle.com/datasets/geekypriyanka/zero-shot-datase) – curated images for zero-shot object detection experiments.
- **Other Options:**

  - [COCO 2017](https://cocodataset.org) – general-purpose object detection.
  - [Open Images V6](https://storage.googleapis.com/openimages/web/index.html) – diverse categories.

- **Model Reference:** [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- **Reference Paper:** Liu et al., _Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection_, CVPR 2023 \[[arXiv](https://arxiv.org/abs/2303.05499)]

---

## 7. Quick Start

### Local / Colab

```bash
# Clone and install
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -r requirements.txt

# Download pretrained checkpoint (example: Swin-B)
wget <https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth>

# Run inference
python demo.py \
  --config configs/GroundingDINO_SwinB.cfg.py \
  --checkpoint groundingdino_swinb.pth \
  --input_image path/to/image.jpg \
  --text_prompt "dog, bottle, tumor"
```

### Kaggle

If you’re running this on **Kaggle Notebooks**, you can directly pull the dataset:

```bash
!kaggle datasets download -d geekypriyanka/zero-shot-datase
unzip zero-shot-datase.zip -d ./data/
```

---

## 8. Project Structure

```
project-root/
├─ zero_shot_pipeline.ipynb   # Main pipeline notebook
├─ README.md                  # Documentation
├─ requirements.txt           # Dependencies
├─ configs/                   # Model configs
├─ data/                      # Sample images / annotations
├─ outputs/                   # Detection results + visualizations
```

---

## 9. Dependencies

See `requirements.txt`. Key packages:

- `torch`
- `transformers`
- `pycocotools`
- `numpy`, `pandas`
- `matplotlib`, `Pillow`

---

## 10. Notes & Recommendations

- Ensure GPU runtime (CUDA) for efficient inference.
- Keep prompt design consistent when comparing results.
- Explore multilingual prompts for broader applications.
- Analyze **failure cases** (missed detections, false positives) to refine insights.

---
