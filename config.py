"""Configuration, model catalog, skeleton topologies, and color palettes for VLM Run Gateway Live Stream App."""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Model Catalog (All 8 VLM Run Gateway Models)
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "rfdetr-large": {
        "id": "rfdetr-large",
        "canonical_id": "roboflow/rfdetr-large",
        "name": "RF-DETR Large",
        "category": "Object Detection",
        "badge": "Detection • 20 FPS",
        "methods": ["detect"],
        "default_method": "detect",
        "needs_prompt": False,
        "default_prompt": "",
        "prompt_placeholder": "No prompt needed for COCO object detection",
        "description": "Real-time Transformer object detection (DINOv2 backbone). Returns normalized bounding boxes, COCO class labels, and confidence scores per frame.",
        "output_type": "img.detect.bboxes",
    },
    "rfdetr-seg-large": {
        "id": "rfdetr-seg-large",
        "canonical_id": "roboflow/rfdetr-seg-large",
        "name": "RF-DETR Seg Large",
        "category": "Instance Segmentation",
        "badge": "Segmentation • 20 FPS",
        "methods": ["segment", "detect"],
        "default_method": "segment",
        "needs_prompt": False,
        "default_prompt": "",
        "prompt_placeholder": "No prompt needed for COCO instance segmentation",
        "description": "Real-time instance segmentation and detection. Emits per-instance boxes, scores, areas, and an 8-bit PNG pixel label map.",
        "output_type": "img.segment.masks",
    },
    "sam3.1": {
        "id": "sam3.1",
        "canonical_id": "facebook/sam3.1",
        "name": "Meta SAM 3.1",
        "category": "Zero-Shot Segmentation & Tracking",
        "badge": "Promptable Masks • 20 FPS",
        "methods": ["segment", "track", "segment_box", "segment_point"],
        "default_method": "segment",
        "needs_prompt": True,
        "default_prompt": "person",
        "prompt_placeholder": "Enter object to segment/track (e.g., person, face, laptop, phone, cup)...",
        "description": "Meta Segment Anything 3.1 with Object Multiplex tracking. Segments or tracks arbitrary objects from open-vocabulary text prompts, boxes, or points.",
        "output_type": "img.segment.masks",
    },
    "vitpose-plus-large": {
        "id": "vitpose-plus-large",
        "canonical_id": "usyd-community/vitpose-plus-large",
        "name": "ViTPose+ Large",
        "category": "2D Human Body Pose",
        "badge": "17 COCO Joints • 20 FPS",
        "methods": ["pose"],
        "default_method": "pose",
        "needs_prompt": False,
        "default_prompt": "",
        "prompt_placeholder": "No prompt needed for human body pose estimation",
        "description": "Large Vision Transformer (434M) for 2D human pose estimation. Detects person boxes and 17 COCO anatomical keypoints with confidence scores.",
        "output_type": "img.pose.kpts",
    },
    "hamer": {
        "id": "hamer",
        "canonical_id": "geopavlakos/hamer",
        "name": "HaMeR (Hand Pose 2D/3D)",
        "category": "3D Hand Mesh & Pose",
        "badge": "21 Hand Joints • 20 FPS",
        "methods": ["pose"],
        "default_method": "pose",
        "needs_prompt": False,
        "default_prompt": "",
        "prompt_placeholder": "No prompt needed for hand tracking",
        "description": "Transformer-based Hand Mesh Recovery. Reconstructs left/right hand detections with 21 normalized 2D keypoints and metric 3D joints.",
        "output_type": "world.pose.kpts",
    },
    "pp-ocrv6": {
        "id": "pp-ocrv6",
        "canonical_id": "paddleocr/pp-ocrv6",
        "name": "PaddleOCR PP-OCRv6",
        "category": "Scene & Document OCR",
        "badge": "OCR & Polygons • 20 FPS",
        "methods": ["ocr", "detect", "text"],
        "default_method": "ocr",
        "needs_prompt": False,
        "default_prompt": "",
        "prompt_placeholder": "Optional language code (e.g. en) is configured in OCR settings",
        "description": "High-speed scene and document text recognition. Extracts rotated polygon outlines, bounding boxes, recognized text lines, and confidence scores.",
        "output_type": "doc.ocr.lines",
    },
    "gemma-4-e2b-it-realtime": {
        "id": "gemma-4-e2b-it-realtime",
        "canonical_id": "google/gemma4-e2b-it:realtime",
        "name": "Gemma 4 E2B IT (Realtime)",
        "category": "Realtime Multimodal VLM",
        "badge": "Live VLM Chat",
        "methods": ["chat"],
        "default_method": "chat",
        "needs_prompt": True,
        "default_prompt": "Describe what is happening in this video frame in one concise sentence.",
        "prompt_placeholder": "Ask anything about the live video feed...",
        "description": "Google DeepMind Gemma 4 Effective-2B instruction-tuned multimodal model optimized for low-latency live visual question answering.",
        "output_type": "chat.text",
    },
    "qwen3.5-0.8b-realtime": {
        "id": "qwen3.5-0.8b-realtime",
        "canonical_id": "qwen/qwen3.5-0.8b:realtime",
        "name": "Qwen 3.5 0.8B (Realtime)",
        "category": "Realtime Multimodal VLM",
        "badge": "Ultra-Fast VLM Chat",
        "methods": ["chat"],
        "default_method": "chat",
        "needs_prompt": True,
        "default_prompt": "List the main objects and actions visible in this frame briefly.",
        "prompt_placeholder": "Ask anything about the live video feed...",
        "description": "Ultra-compact 0.8B dense Vision-Language Model optimized for high-throughput, low-latency real-time visual understanding.",
        "output_type": "chat.text",
    },
}

MODEL_CHOICES: list[tuple[str, str]] = [
    (f"{info['name']}  —  [{info['category']}]", key)
    for key, info in MODEL_CATALOG.items()
]


# ---------------------------------------------------------------------------
# Color Palettes (RGB format for Gradio / PIL / NumPy)
# ---------------------------------------------------------------------------

PALETTE_RGB: list[tuple[int, int, int]] = [
    (56, 189, 248),   # Sky Blue
    (168, 85, 247),   # Purple
    (52, 211, 153),   # Emerald
    (251, 146, 60),   # Orange
    (244, 114, 182),  # Pink
    (250, 204, 21),   # Yellow
    (96, 165, 250),   # Blue
    (248, 113, 113),  # Red
    (45, 212, 191),   # Teal
    (192, 132, 252),  # Lavender
    (163, 230, 53),   # Lime
    (251, 113, 133),  # Rose
]


def get_color_rgb(index_or_label: int | str) -> tuple[int, int, int]:
    """Return a deterministic vivid RGB color for a class label or instance ID."""
    if isinstance(index_or_label, int):
        return PALETTE_RGB[index_or_label % len(PALETTE_RGB)]
    h = sum(ord(c) * (i + 1) for i, c in enumerate(str(index_or_label)))
    return PALETTE_RGB[h % len(PALETTE_RGB)]


# ---------------------------------------------------------------------------
# Pose Skeleton Topologies
# ---------------------------------------------------------------------------

# 17 COCO body keypoints (usyd-community/vitpose-plus-large)
COCO_17_SKELETON: list[tuple[int, int, tuple[int, int, int]]] = [
    # Head / Face (Sky Blue)
    (0, 1, (56, 189, 248)),
    (0, 2, (56, 189, 248)),
    (1, 3, (56, 189, 248)),
    (2, 4, (56, 189, 248)),
    # Shoulders & Torso (Emerald Green)
    (5, 6, (52, 211, 153)),
    (5, 11, (52, 211, 153)),
    (6, 12, (52, 211, 153)),
    (11, 12, (52, 211, 153)),
    # Left Arm (Orange)
    (5, 7, (251, 146, 60)),
    (7, 9, (251, 146, 60)),
    # Right Arm (Pink)
    (6, 8, (244, 114, 182)),
    (8, 10, (244, 114, 182)),
    # Left Leg (Yellow)
    (11, 13, (250, 204, 21)),
    (13, 15, (250, 204, 21)),
    # Right Leg (Purple)
    (12, 14, (168, 85, 247)),
    (14, 16, (168, 85, 247)),
]

# 21 Hand keypoints (geopavlakos/hamer)
HAND_21_SKELETON: list[tuple[int, int, tuple[int, int, int]]] = [
    # Thumb (Orange)
    (0, 1, (251, 146, 60)),
    (1, 2, (251, 146, 60)),
    (2, 3, (251, 146, 60)),
    (3, 4, (251, 146, 60)),
    # Index (Sky Blue)
    (0, 5, (56, 189, 248)),
    (5, 6, (56, 189, 248)),
    (6, 7, (56, 189, 248)),
    (7, 8, (56, 189, 248)),
    # Middle (Emerald)
    (0, 9, (52, 211, 153)),
    (9, 10, (52, 211, 153)),
    (10, 11, (52, 211, 153)),
    (11, 12, (52, 211, 153)),
    # Ring (Purple)
    (0, 13, (168, 85, 247)),
    (13, 14, (168, 85, 247)),
    (14, 15, (168, 85, 247)),
    (15, 16, (168, 85, 247)),
    # Pinky (Pink)
    (0, 17, (244, 114, 182)),
    (17, 18, (244, 114, 182)),
    (18, 19, (244, 114, 182)),
    (19, 20, (244, 114, 182)),
]
