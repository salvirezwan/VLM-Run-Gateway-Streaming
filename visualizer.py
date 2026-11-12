"""Fast OpenCV / NumPy visualization engine for VLM Run Gateway responses."""

from __future__ import annotations

import base64
import textwrap
from typing import Any

import cv2
import numpy as np

from config import COCO_17_SKELETON, HAND_21_SKELETON, get_color_rgb


def _decode_png_mask(mask_obj: dict[str, Any] | None, target_w: int, target_h: int) -> np.ndarray | None:
    """Decode base64 8-bit PNG label map into a 2D uint8 numpy array of shape (target_h, target_w)."""
    if not mask_obj or not isinstance(mask_obj, dict):
        return None
    data_uri = mask_obj.get("data", "")
    if not data_uri:
        return None
    try:
        if "," in data_uri:
            b64_str = data_uri.split(",", 1)[1]
        else:
            b64_str = data_uri
        raw_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        label_map = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if label_map is None:
            return None
        if label_map.ndim == 3:
            label_map = label_map[:, :, 0]
        if label_map.shape[0] != target_h or label_map.shape[1] != target_w:
            label_map = cv2.resize(label_map, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return label_map
    except Exception:
        return None


def _draw_label_badge(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color_rgb: tuple[int, int, int],
    font_scale: float = 0.52,
    thickness: int = 1,
) -> None:
    """Draw a high-contrast pill badge with text at (x, y)."""
    if not text:
        return
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 7, 5
    box_w = tw + pad_x * 2
    box_h = th + baseline + pad_y * 2

    bx1 = max(2, min(x, w - box_w - 2))
    by1 = max(2, y - box_h if y - box_h >= 2 else y + 2)
    bx2 = min(w - 2, bx1 + box_w)
    by2 = min(h - 2, by1 + box_h)

    # Dark translucent background with colored left accent bar
    roi = img[by1:by2, bx1:bx2]
    bg = np.full_like(roi, (18, 22, 32), dtype=np.uint8)
    cv2.addWeighted(bg, 0.82, roi, 0.18, 0, roi)
    img[by1:by2, bx1:bx2] = roi

    # Colored border and left bar
    cv2.rectangle(img, (bx1, by1), (bx2, by2), color_rgb, 1, cv2.LINE_AA)
    cv2.rectangle(img, (bx1, by1), (min(bx2, bx1 + 4), by2), color_rgb, -1)

    # Crisp white text
    tx = bx1 + pad_x + 2
    ty = by1 + pad_y + th
    cv2.putText(img, text, (tx, ty), font, font_scale, (248, 250, 252), thickness, cv2.LINE_AA)


def _draw_fancy_bbox(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_rgb: tuple[int, int, int],
    fill_alpha: float = 0.10,
) -> None:
    """Draw a modern bounding box with subtle tint and HUD corner accents."""
    h, w = img.shape[:2]
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    if fill_alpha > 0:
        roi = img[y1:y2, x1:x2]
        color_patch = np.full_like(roi, color_rgb, dtype=np.uint8)
        cv2.addWeighted(color_patch, fill_alpha, roi, 1.0 - fill_alpha, 0, roi)
        img[y1:y2, x1:x2] = roi

    # Main thin rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, 1, cv2.LINE_AA)

    # Thick corner brackets
    bw, bh = x2 - x1, y2 - y1
    clen = max(6, min(18, int(min(bw, bh) * 0.22)))
    t = 2
    # Top-left
    cv2.line(img, (x1, y1), (x1 + clen, y1), color_rgb, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + clen), color_rgb, t, cv2.LINE_AA)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - clen, y1), color_rgb, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + clen), color_rgb, t, cv2.LINE_AA)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + clen, y2), color_rgb, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - clen), color_rgb, t, cv2.LINE_AA)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - clen, y2), color_rgb, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - clen), color_rgb, t, cv2.LINE_AA)


def _draw_hud_overlay(
    img: np.ndarray,
    model_name: str,
    method: str,
    transport: str,
    latency_ms: float | None,
    count_str: str,
    status_note: str = "",
) -> None:
    """Draw a sleek top-left telemetry bar on the video frame."""
    h, w = img.shape[:2]
    lat_part = f"{latency_ms:.0f} ms" if latency_ms is not None else "-- ms"
    parts = [f"{transport.upper()}", f"{model_name} ({method})", lat_part]
    if count_str:
        parts.append(count_str)
    if status_note:
        parts.append(status_note)
    banner_text = "  |  ".join(parts)

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.48
    (tw, th), baseline = cv2.getTextSize(banner_text, font, scale, 1)
    pad_x, pad_y = 12, 8
    bx1, by1 = 10, 10
    bx2 = min(w - 10, bx1 + tw + pad_x * 2 + 14)
    by2 = min(h - 10, by1 + th + baseline + pad_y * 2)

    roi = img[by1:by2, bx1:bx2]
    bg = np.full_like(roi, (15, 23, 42), dtype=np.uint8)
    cv2.addWeighted(bg, 0.80, roi, 0.20, 0, roi)
    img[by1:by2, bx1:bx2] = roi
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (56, 189, 248), 1, cv2.LINE_AA)

    # Live status dot
    dot_color = (52, 211, 153) if not status_note else (250, 204, 21)
    cy = (by1 + by2) // 2
    cv2.circle(img, (bx1 + 12, cy), 4, dot_color, -1, cv2.LINE_AA)

    cv2.putText(
        img,
        banner_text,
        (bx1 + 24, by1 + pad_y + th),
        font,
        scale,
        (241, 245, 249),
        1,
        cv2.LINE_AA,
    )


def _draw_bottom_caption_hud(img: np.ndarray, text: str) -> None:
    """Draw a readable bottom subtitle overlay for VLM chat / plain text responses."""
    clean = " ".join(text.strip().split())
    if not clean:
        return
    h, w = img.shape[:2]
    max_chars = max(35, int(w / 11))
    lines = textwrap.wrap(clean, width=max_chars)[:4]

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.54
    line_h = 22
    pad_x, pad_y = 16, 12
    box_h = len(lines) * line_h + pad_y * 2
    by2 = h - 12
    by1 = max(50, by2 - box_h)
    bx1 = 14
    bx2 = w - 14

    roi = img[by1:by2, bx1:bx2]
    bg = np.full_like(roi, (15, 23, 42), dtype=np.uint8)
    cv2.addWeighted(bg, 0.82, roi, 0.18, 0, roi)
    img[by1:by2, bx1:bx2] = roi
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (168, 85, 247), 1, cv2.LINE_AA)

    for idx, line in enumerate(lines):
        ty = by1 + pad_y + 16 + idx * line_h
        cv2.putText(img, line, (bx1 + pad_x, ty), font, scale, (248, 250, 252), 1, cv2.LINE_AA)


def render_predictions(
    frame_rgb: np.ndarray,
    parsed_payload: dict[str, Any] | str | None,
    *,
    model_name: str,
    method: str,
    transport: str = "WS",
    latency_ms: float | None = None,
    conf_threshold: float = 0.25,
    show_labels: bool = True,
    status_note: str = "",
) -> tuple[np.ndarray, str]:
    """
    Render VLM Run Gateway predictions onto an RGB numpy frame.

    Returns:
        (annotated_rgb_frame, formatted_markdown_summary)
    """
    canvas = frame_rgb.copy()
    h, w = canvas.shape[:2]

    if parsed_payload is None:
        _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, "Waiting...", status_note)
        return canvas, "_Waiting for first prediction..._"

    # Unwrap JSON-mode envelope if present: {"model": ..., "method": ..., "content": ...}
    content: Any = parsed_payload
    if isinstance(parsed_payload, dict) and "content" in parsed_payload and isinstance(
        parsed_payload.get("content"), (dict, str)
    ):
        content = parsed_payload["content"]

    # Case 1: Plain text or VLM chat string
    if isinstance(content, str):
        _draw_bottom_caption_hud(canvas, content)
        _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, "Text reply", status_note)
        return canvas, f"### Live Model Output\n\n{content}"

    if not isinstance(content, dict):
        text_repr = str(content)
        _draw_bottom_caption_hud(canvas, text_repr)
        _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, "Reply", status_note)
        return canvas, f"```json\n{text_repr}\n```"

    obj_type = content.get("object", "")
    items = content.get("items", [])
    if not isinstance(items, list):
        items = []

    summary_lines: list[str] = []

    # -----------------------------------------------------------------------
    # 1. Object Detection: img.detect.bboxes
    # -----------------------------------------------------------------------
    if obj_type == "img.detect.bboxes":
        kept = 0
        for idx, item in enumerate(items):
            score = item.get("score")
            if score is not None and score < conf_threshold:
                continue
            bbox = item.get("bbox_xywh")
            if not bbox or len(bbox) != 4:
                continue
            kept += 1
            bx, by, bw, bh = bbox
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)
            label = item.get("label") or "object"
            color = get_color_rgb(label)
            _draw_fancy_bbox(canvas, x1, y1, x2, y2, color, fill_alpha=0.12)
            badge = f"{label} {score:.0%}" if score is not None else label
            if show_labels:
                _draw_label_badge(canvas, badge, x1, y1, color)
            summary_lines.append(f"- **{label}**" + (f" (`{score:.1%}`)" if score is not None else ""))

        _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, f"{kept} objects", status_note)
        header = f"### Detected Objects ({kept})\n"
        return canvas, header + ("\n".join(summary_lines[:30]) if summary_lines else "_No objects above threshold._")

    # -----------------------------------------------------------------------
    # 2. Instance / Video Segmentation: img.segment.masks / vid.segment.masks
    # -----------------------------------------------------------------------
    if obj_type in ("img.segment.masks", "vid.segment.masks"):
        mask_obj = content.get("mask")
        if not mask_obj and obj_type == "vid.segment.masks":
            frames_list = content.get("frames", [])
            if frames_list and isinstance(frames_list[-1], dict):
                mask_obj = frames_list[-1].get("mask")

        label_map = _decode_png_mask(mask_obj, w, h)
        kept = 0

        # Blend masks first
        if label_map is not None:
            overlay = canvas.copy()
            active_mask = np.zeros((h, w), dtype=bool)
            for idx, item in enumerate(items):
                score = item.get("score")
                if score is not None and score < conf_threshold:
                    continue
                inst_id = item.get("instance_id") or item.get("track_id") or (idx + 1)
                color = get_color_rgb(int(inst_id))
                m = label_map == int(inst_id)
                if np.any(m):
                    overlay[m] = color
                    active_mask |= m
                    # Draw crisp contour around the instance mask
                    m_u8 = (m.astype(np.uint8)) * 255
                    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)

            if np.any(active_mask):
                blended = cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0)
                canvas[active_mask] = blended[active_mask]

        # Draw polygons (if present) + bounding boxes + badges
        for idx, item in enumerate(items):
            score = item.get("score")
            if score is not None and score < conf_threshold:
                continue
            kept += 1
            inst_id = item.get("instance_id") or item.get("track_id") or (idx + 1)
            label = item.get("label") or "instance"
            area = item.get("area")
            color = get_color_rgb(int(inst_id))

            polys = item.get("polys_xy")
            if polys:
                for ring in polys:
                    pts = np.array([[int(px * w), int(py * h)] for px, py in ring], dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)

            bbox = item.get("bbox_xywh")
            if bbox and len(bbox) == 4:
                bx, by, bw, bh = bbox
                x1, y1 = int(bx * w), int(by * h)
                x2, y2 = int((bx + bw) * w), int((by + bh) * h)
                _draw_fancy_bbox(canvas, x1, y1, x2, y2, color, fill_alpha=0.0)
                badge_parts = [f"{label} #{inst_id}"]
                if score is not None:
                    badge_parts.append(f"{score:.0%}")
                if show_labels:
                    _draw_label_badge(canvas, " ".join(badge_parts), x1, y1, color)

            meta = []
            if score is not None:
                meta.append(f"score `{score:.1%}`")
            if area is not None:
                meta.append(f"area `{area:.1%}`")
            summary_lines.append(f"- **{label} #{inst_id}**" + (f" ({', '.join(meta)})" if meta else ""))

        _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, f"{kept} masks", status_note)
        header = f"### Segmented Instances ({kept})\n"
        return canvas, header + ("\n".join(summary_lines[:30]) if summary_lines else "_No instances segmented._")

    # -----------------------------------------------------------------------
    # 3. OCR Lines & Boxes: doc.ocr.lines / doc.ocr.bboxes / doc.page.blocks
    # -----------------------------------------------------------------------
    if obj_type in ("doc.ocr.lines", "doc.ocr.bboxes", "doc.page.blocks", "doc.layout.bboxes"):
        kept = 0
        for idx, item in enumerate(items):
            score = item.get("score")
            if score is not None and score < conf_threshold:
                continue
            kept += 1
            text_val = item.get("text", "")
            label = item.get("label") or ""
            color = get_color_rgb(label if label else idx)

            poly = item.get("poly_xy")
            bbox = item.get("bbox_xywh")
            anchor_x, anchor_y = 10, 30

            if poly and len(poly) >= 3:
                pts = np.array([[int(px * w), int(py * h)] for px, py in poly], dtype=np.int32)
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)
                cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
                anchor_x, anchor_y = int(pts[0][0]), int(pts[0][1])
            elif bbox and len(bbox) == 4:
                bx, by, bw, bh = bbox
                x1, y1 = int(bx * w), int(by * h)
                x2, y2 = int((bx + bw) * w), int((by + bh) * h)
                _draw_fancy_bbox(canvas, x1, y1, x2, y2, color, fill_alpha=0.14)
                anchor_x, anchor_y = x1, y1

            if show_labels:
                disp = text_val if text_val else (label if label else f"region #{idx + 1}")
                if len(disp) > 36:
                    disp = disp[:33] + "..."
                _draw_label_badge(canvas, disp, anchor_x, anchor_y, color, font_scale=0.46)

            if text_val:
                score_str = f" (`{score:.0%}`)" if score is not None else ""
                summary_lines.append(f"{kept}. `{text_val}`{score_str}")
            elif label:
                summary_lines.append(f"- **{label}** region")

        _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, f"{kept} text regions", status_note)
        header = f"### OCR / Layout Regions ({kept})\n"
        return canvas, header + ("\n".join(summary_lines[:40]) if summary_lines else "_No text regions detected._")

    # -----------------------------------------------------------------------
    # 4. Human Body Pose (COCO 17): img.pose.kpts / vid.pose.kpts
    # -----------------------------------------------------------------------
    if obj_type in ("img.pose.kpts", "vid.pose.kpts"):
        kept = 0
        for idx, person in enumerate(items):
            kept += 1
            track_id = person.get("track_id")
            bbox = person.get("bbox_xywh")
            p_color = get_color_rgb(track_id if track_id is not None else idx)

            if bbox and len(bbox) == 4:
                bx, by, bw, bh = bbox
                x1, y1 = int(bx * w), int(by * h)
                x2, y2 = int((bx + bw) * w), int((by + bh) * h)
                _draw_fancy_bbox(canvas, x1, y1, x2, y2, p_color, fill_alpha=0.05)
                if show_labels:
                    tag = f"Person #{track_id}" if track_id is not None else f"Person #{idx + 1}"
                    _draw_label_badge(canvas, tag, x1, y1, p_color)

            kpts_xy = person.get("kpts_xy", [])
            kpts_score = person.get("kpts_score", [])
            num_kpts = len(kpts_xy)

            # Draw skeleton limbs
            for i_a, i_b, limb_color in COCO_17_SKELETON:
                if i_a < num_kpts and i_b < num_kpts:
                    sa = kpts_score[i_a] if i_a < len(kpts_score) else 1.0
                    sb = kpts_score[i_b] if i_b < len(kpts_score) else 1.0
                    if sa >= conf_threshold and sb >= conf_threshold:
                        pt_a = (int(kpts_xy[i_a][0] * w), int(kpts_xy[i_a][1] * h))
                        pt_b = (int(kpts_xy[i_b][0] * w), int(kpts_xy[i_b][1] * h))
                        cv2.line(canvas, pt_a, pt_b, limb_color, 3, cv2.LINE_AA)

            # Draw joints
            vis_joints = 0
            for j_idx, xy in enumerate(kpts_xy):
                sc = kpts_score[j_idx] if j_idx < len(kpts_score) else 1.0
                if sc >= conf_threshold:
                    vis_joints += 1
                    px, py = int(xy[0] * w), int(xy[1] * h)
                    cv2.circle(canvas, (px, py), 5, (15, 23, 42), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 4, (56, 189, 248), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

            summary_lines.append(f"- **Person #{idx + 1}**: `{vis_joints}/{num_kpts}` visible keypoints")

        _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, f"{kept} people", status_note)
        header = f"### Body Pose Estimation ({kept} People)\n"
        return canvas, header + ("\n".join(summary_lines) if summary_lines else "_No people detected._")

    # -----------------------------------------------------------------------
    # 5. Hand Pose 2D & 3D (HaMeR): world.pose.kpts
    # -----------------------------------------------------------------------
    if obj_type == "world.pose.kpts":
        kept = 0
        for idx, hand in enumerate(items):
            score = hand.get("score")
            if score is not None and score < conf_threshold:
                continue
            kept += 1
            label = hand.get("label", "hand")
            pretty_label = "Left Hand" if "left" in label.lower() else ("Right Hand" if "right" in label.lower() else label)
            h_color = (56, 189, 248) if "left" in label.lower() else (244, 114, 182)

            bbox = hand.get("bbox_xywh")
            if bbox and len(bbox) == 4:
                bx, by, bw, bh = bbox
                x1, y1 = int(bx * w), int(by * h)
                x2, y2 = int((bx + bw) * w), int((by + bh) * h)
                _draw_fancy_bbox(canvas, x1, y1, x2, y2, h_color, fill_alpha=0.07)
                if show_labels:
                    badge = f"{pretty_label} {score:.0%}" if score is not None else pretty_label
                    _draw_label_badge(canvas, badge, x1, y1, h_color)

            kpts_xy = hand.get("kpts_xy", [])
            kpts_xyz = hand.get("kpts_xyz", [])
            num_kpts = len(kpts_xy)

            # Draw finger bones
            for i_a, i_b, bone_color in HAND_21_SKELETON:
                if i_a < num_kpts and i_b < num_kpts:
                    pt_a = (int(kpts_xy[i_a][0] * w), int(kpts_xy[i_a][1] * h))
                    pt_b = (int(kpts_xy[i_b][0] * w), int(kpts_xy[i_b][1] * h))
                    cv2.line(canvas, pt_a, pt_b, bone_color, 3, cv2.LINE_AA)

            # Draw 21 hand joints
            for j_idx, xy in enumerate(kpts_xy):
                px, py = int(xy[0] * w), int(xy[1] * h)
                cv2.circle(canvas, (px, py), 5, (15, 23, 42), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)

            xyz_info = ""
            if kpts_xyz and len(kpts_xyz) > 8 and len(kpts_xyz[0]) == 3:
                wx, wy, wz = kpts_xyz[0]
                ix, iy, iz = kpts_xyz[8]  # index_tip
                xyz_info = f" — 3D wrist `({wx:+.2f}, {wy:+.2f}, {wz:+.2f})m`, index tip `({ix:+.2f}, {iy:+.2f}, {iz:+.2f})m`"
            summary_lines.append(
                f"- **{pretty_label}**"
                + (f" (`{score:.1%}`)" if score is not None else "")
                + f": `{num_kpts}` 2D/3D joints{xyz_info}"
            )

        _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, f"{kept} hands", status_note)
        header = f"### 3D Hand Pose Recovery ({kept} Hands)\n"
        return canvas, header + ("\n".join(summary_lines) if summary_lines else "_No hands detected._")

    # Fallback for any other JSON structure
    _draw_hud_overlay(canvas, model_name, method, transport, latency_ms, obj_type or "JSON", status_note)
    return canvas, f"### Gateway Response (`{obj_type or 'json'}`)\n\n```json\n{content}\n```"
