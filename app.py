"""VLM Run Gateway — Live Video Detection & Visual Intelligence Gradio App."""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import cv2
import gradio as gr
import numpy as np

from config import MODEL_CATALOG, MODEL_CHOICES
from gateway_client import (
    ENV_CONFIG,
    GatewayLiveStreamer,
    StreamSessionConfig,
    TelemetryState,
    derive_ws_url,
    run_single_frame_test,
)
from visualizer import render_predictions

# Global background streamer instance
LIVE_STREAMER = GatewayLiveStreamer()


# ---------------------------------------------------------------------------
# HTML Builders for Rich UI Components
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --vlm-bg-card: #0f172a;
    --vlm-bg-elevated: #1e293b;
    --vlm-border: #334155;
    --vlm-accent: #38bdf8;
    --vlm-accent-purple: #a855f7;
    --vlm-success: #34d399;
    --vlm-warning: #fbbf24;
    --vlm-danger: #f87171;
}

.gradio-container {
    max-width: 1560px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.vlm-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 55%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 26px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.vlm-header-left h1 {
    margin: 0 !important;
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 55%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

.vlm-header-left p {
    margin: 5px 0 0 0 !important;
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
}

.vlm-badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
}

.vlm-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

.vlm-pill.purple {
    background: rgba(168, 85, 247, 0.12);
    color: #c084fc;
    border-color: rgba(168, 85, 247, 0.3);
}

.vlm-pill.green {
    background: rgba(52, 211, 153, 0.12);
    color: #34d399;
    border-color: rgba(52, 211, 153, 0.3);
}

.vlm-model-card {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 13px 16px;
    color: #e2e8f0;
}

.vlm-model-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}

.vlm-model-title {
    font-weight: 700;
    font-size: 1.0rem;
    color: #f8fafc;
}

.vlm-model-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
    font-size: 0.76rem;
}

.vlm-code-tag {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid #475569;
    color: #93c5fd;
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 0.74rem;
}

.vlm-telemetry-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 6px;
}

@media (max-width: 900px) {
    .vlm-telemetry-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

.vlm-stat-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 13px;
}

.vlm-stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 3px;
}

.vlm-stat-value {
    font-size: 1.08rem;
    font-weight: 700;
    color: #f8fafc;
    font-family: 'JetBrains Mono', Consolas, monospace;
}

.vlm-stat-sub {
    font-size: 0.73rem;
    color: #64748b;
    margin-top: 2px;
}

.vlm-status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 14px;
    border-radius: 10px;
    background: #0f172a;
    border: 1px solid #334155;
    margin-bottom: 8px;
    font-size: 0.84rem;
    color: #e2e8f0;
}

.vlm-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

.vlm-dot.ready { background: #38bdf8; box-shadow: 0 0 8px #38bdf8; }
.vlm-dot.streaming { background: #34d399; box-shadow: 0 0 10px #34d399; }
.vlm-dot.provisioning, .vlm-dot.fallback { background: #fbbf24; box-shadow: 0 0 10px #fbbf24; }
.vlm-dot.error { background: #f87171; box-shadow: 0 0 10px #f87171; }
"""


def render_header_html(env_choice: str = "dev") -> str:
    base_url = (
        ENV_CONFIG["prod_base_url"]
        if env_choice == "prod"
        else ENV_CONFIG["base_url"]
    )
    ws_url = derive_ws_url(base_url)
    return f"""
    <div class="vlm-header">
        <div class="vlm-header-left">
            <h1>VLM Run Gateway — Realtime Visual Intelligence Studio</h1>
            <p>Live webcam detection, zero-shot segmentation, 2D/3D pose recovery, OCR, and multimodal VLM streaming</p>
        </div>
        <div class="vlm-badge-row">
            <span class="vlm-pill green">● WS: {ws_url}</span>
            <span class="vlm-pill">HTTP Fallback Enabled</span>
            <span class="vlm-pill purple">8 Gateway Models Ready</span>
        </div>
    </div>
    """


def render_model_card_html(model_key: str) -> str:
    info = MODEL_CATALOG.get(model_key, MODEL_CATALOG["rfdetr-large"])
    methods_str = ", ".join(info["methods"])
    prompt_tag = "Prompt-Driven" if info["needs_prompt"] else "Zero-Config Feed"
    return f"""
    <div class="vlm-model-card">
        <div class="vlm-model-card-top">
            <span class="vlm-model-title">{info['name']}</span>
            <span class="vlm-pill">{info['badge']}</span>
        </div>
        <div style="font-size: 0.83rem; color: #cbd5e1; line-height: 1.45;">
            {info['description']}
        </div>
        <div class="vlm-model-meta">
            <span class="vlm-code-tag">id: {info['canonical_id']}</span>
            <span class="vlm-code-tag">methods: {methods_str}</span>
            <span class="vlm-code-tag">payload: {info['output_type']}</span>
            <span class="vlm-code-tag">{prompt_tag}</span>
        </div>
    </div>
    """


def render_telemetry_html(t: TelemetryState) -> str:
    lat_str = f"{t.last_latency_ms:.0f} ms" if t.last_latency_ms is not None else "—"
    fps_str = f"{t.ema_fps:.1f} FPS" if t.ema_fps > 0 else f"Max {t.max_fps:.0f} FPS"
    state_cls = t.status_state if t.status_state in ("ready", "streaming", "provisioning", "fallback", "error") else "ready"
    return f"""
    <div>
        <div class="vlm-status-bar">
            <span class="vlm-dot {state_cls}"></span>
            <span style="font-weight: 600;">Status:</span>
            <span style="color: #cbd5e1;">{t.status_message}</span>
        </div>
        <div class="vlm-telemetry-grid">
            <div class="vlm-stat-card">
                <div class="vlm-stat-label">Active Transport</div>
                <div class="vlm-stat-value">{t.active_transport}</div>
                <div class="vlm-stat-sub">Session: {t.session_id}</div>
            </div>
            <div class="vlm-stat-card">
                <div class="vlm-stat-label">Read Latency</div>
                <div class="vlm-stat-value">{lat_str}</div>
                <div class="vlm-stat-sub">Throughput: {fps_str}</div>
            </div>
            <div class="vlm-stat-card">
                <div class="vlm-stat-label">Frames (Read / Sent)</div>
                <div class="vlm-stat-value">{t.frames_read} / {t.frames_sent}</div>
                <div class="vlm-stat-sub">Dropped (backpressure): {t.frames_dropped}</div>
            </div>
            <div class="vlm-stat-card">
                <div class="vlm-stat-label">Target Model</div>
                <div class="vlm-stat-value" style="font-size: 0.92rem;">{t.canonical_model}</div>
                <div class="vlm-stat-sub">Errors: {t.errors_count} • Cost: ${t.session_cost_usd:.4f}</div>
            </div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Callbacks & Stream Handlers
# ---------------------------------------------------------------------------

def on_model_change(model_key: str) -> tuple[Any, Any, str]:
    """Update method dropdown choices, prompt box, and model info card when model changes."""
    info = MODEL_CATALOG.get(model_key, MODEL_CATALOG["rfdetr-large"])
    methods = info["methods"]
    default_method = info["default_method"]
    default_prompt = info["default_prompt"]
    placeholder = info["prompt_placeholder"]

    method_update = gr.update(choices=methods, value=default_method)
    prompt_update = gr.update(
        value=default_prompt,
        placeholder=placeholder,
        interactive=True,
        label=(
            "Target Prompt (Required for SAM 3.1 & Realtime VLMs)"
            if info["needs_prompt"]
            else "Optional Prompt (Used by SAM 3.1 & Realtime VLMs)"
        ),
    )
    card_html = render_model_card_html(model_key)
    return method_update, prompt_update, card_html


def process_webcam_stream(
    frame: np.ndarray | None,
    model_key: str,
    method: str,
    prompt: str,
    transport_mode: str,
    env_choice: str,
    conf_threshold: float,
    show_labels: bool,
    max_inflight: int,
    max_dimension: int,
    ocr_lang: str,
    ocr_word_boxes: bool,
) -> tuple[np.ndarray | None, str, str, dict[str, Any]]:
    """
    Non-blocking Gradio webcam stream callback:
    1. Updates active session configuration on the background worker.
    2. Submits the latest webcam frame to the WebSocket / HTTP fallback worker.
    3. Immediately overlays the most recent Gateway prediction onto the current live frame.
    """
    if frame is None:
        snap = LIVE_STREAMER.get_snapshot()
        return None, render_telemetry_html(snap), "_Waiting for camera stream..._", snap.latest_raw_response

    info = MODEL_CATALOG.get(model_key, MODEL_CATALOG["rfdetr-large"])
    valid_method = method if method in info["methods"] else info["default_method"]

    cfg = StreamSessionConfig(
        model=model_key,
        method=valid_method,
        prompt=prompt or info["default_prompt"],
        transport_mode=transport_mode,
        environment=env_choice,
        max_inflight=int(max_inflight),
        max_dimension=int(max_dimension),
        ocr_lang=ocr_lang,
        ocr_word_boxes=bool(ocr_word_boxes),
    )
    LIVE_STREAMER.update_config(cfg)
    LIVE_STREAMER.submit_frame(frame)

    snap = LIVE_STREAMER.get_snapshot()
    status_note = ""
    if snap.status_state == "provisioning":
        status_note = "Provisioning replica..."
    elif snap.status_state == "error":
        status_note = "Check status bar"

    annotated, summary_md = render_predictions(
        frame,
        snap.latest_payload,
        model_name=info["name"],
        method=valid_method,
        transport=snap.active_transport,
        latency_ms=snap.last_latency_ms,
        conf_threshold=float(conf_threshold),
        show_labels=bool(show_labels),
        status_note=status_note,
    )
    return annotated, render_telemetry_html(snap), summary_md, snap.latest_raw_response


def on_stop_stream() -> tuple[str, str]:
    """Close the active WebSocket session cleanly when the user stops streaming."""
    LIVE_STREAMER.stop_session()
    time.sleep(0.15)
    snap = LIVE_STREAMER.get_snapshot()
    return render_telemetry_html(snap), "_Stream stopped. WebSocket session closed cleanly._"


def generate_demo_card_image() -> np.ndarray:
    """Create a rich synthetic test frame (with text, shapes, and person silhouette) for quick testing."""
    img = np.full((540, 860, 3), (22, 30, 46), dtype=np.uint8)
    # Grid lines
    for x in range(0, 860, 40):
        cv2.line(img, (x, 0), (x, 540), (32, 43, 64), 1)
    for y in range(0, 540, 40):
        cv2.line(img, (0, y), (860, y), (32, 43, 64), 1)

    # Receipt / Sign board for OCR
    cv2.rectangle(img, (50, 50), (420, 290), (245, 247, 250), -1)
    cv2.rectangle(img, (50, 50), (420, 290), (56, 189, 248), 3)
    cv2.putText(img, "VLM RUN GATEWAY TEST", (75, 100), cv2.FONT_HERSHEY_DUPLEX, 0.78, (15, 23, 42), 2, cv2.LINE_AA)
    cv2.putText(img, "Model: PP-OCRv6 & SAM 3.1", (75, 145), cv2.FONT_HERSHEY_DUPLEX, 0.62, (30, 41, 59), 1, cv2.LINE_AA)
    cv2.putText(img, "Status: LIVE STREAM READY", (75, 185), cv2.FONT_HERSHEY_DUPLEX, 0.62, (16, 120, 80), 2, cv2.LINE_AA)
    cv2.putText(img, "Invoice #INV-2026-098  $149.50", (75, 230), cv2.FONT_HERSHEY_DUPLEX, 0.58, (30, 41, 59), 1, cv2.LINE_AA)
    cv2.putText(img, "Latency Target: < 250 ms", (75, 268), cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 116, 139), 1, cv2.LINE_AA)

    # Stylized person / object figures on the right
    cv2.circle(img, (630, 140), 42, (251, 191, 36), -1, cv2.LINE_AA)  # Head
    cv2.rectangle(img, (580, 190), (680, 350), (56, 189, 248), -1)    # Torso
    cv2.line(img, (580, 210), (515, 300), (56, 189, 248), 22, cv2.LINE_AA)
    cv2.line(img, (680, 210), (745, 300), (56, 189, 248), 22, cv2.LINE_AA)
    cv2.line(img, (605, 350), (585, 485), (148, 163, 184), 24, cv2.LINE_AA)
    cv2.line(img, (655, 350), (675, 485), (148, 163, 184), 24, cv2.LINE_AA)

    # Cup / Laptop object shapes
    cv2.rectangle(img, (120, 360), (340, 485), (168, 85, 247), -1)
    cv2.putText(img, "LAPTOP / MONITOR", (142, 430), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run_snapshot_inference(
    image: np.ndarray | None,
    model_key: str,
    method: str,
    prompt: str,
    transport_mode: str,
    env_choice: str,
    conf_threshold: float,
    show_labels: bool,
    max_dimension: int,
    ocr_lang: str,
    ocr_word_boxes: bool,
) -> tuple[np.ndarray | None, str, dict[str, Any]]:
    """Run one-shot inference on a still image or snapshot using the selected Gateway model."""
    if image is None:
        image = generate_demo_card_image()

    info = MODEL_CATALOG.get(model_key, MODEL_CATALOG["rfdetr-large"])
    valid_method = method if method in info["methods"] else info["default_method"]

    cfg = StreamSessionConfig(
        model=model_key,
        method=valid_method,
        prompt=prompt or info["default_prompt"],
        transport_mode=transport_mode,
        environment=env_choice,
        max_inflight=1,
        max_dimension=int(max_dimension),
        ocr_lang=ocr_lang,
        ocr_word_boxes=bool(ocr_word_boxes),
    )

    parsed, raw_resp, latency_ms, status_txt = asyncio.run(run_single_frame_test(image, cfg))
    annotated, summary_md = render_predictions(
        image,
        parsed,
        model_name=info["name"],
        method=valid_method,
        transport=status_txt.split()[0] if status_txt else "Gateway",
        latency_ms=latency_ms,
        conf_threshold=float(conf_threshold),
        show_labels=bool(show_labels),
        status_note="" if parsed is not None else status_txt,
    )
    full_md = f"**Status:** `{status_txt}`\n\n---\n\n{summary_md}"
    return annotated, full_md, raw_resp


# ---------------------------------------------------------------------------
# Build Gradio Blocks Application
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    blocks_kwargs: dict[str, Any] = {"title": "VLM Run Gateway — Live Stream Detection"}
    sig = inspect.signature(gr.Blocks.__init__)
    if "theme" in sig.parameters:
        blocks_kwargs["theme"] = gr.themes.Soft(
            primary_hue="sky",
            secondary_hue="purple",
            neutral_hue="slate",
            font=["Inter", "system-ui", "-apple-system", "sans-serif"],
            font_mono=["JetBrains Mono", "Consolas", "monospace"],
        )
    if "css" in sig.parameters:
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        header_html = gr.HTML(render_header_html("dev"))

        # Top Global Configuration Bar
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="rfdetr-large",
                        label="1. VLM Run Gateway Model",
                         scale=3,
                    )
                    method_dropdown = gr.Dropdown(
                        choices=MODEL_CATALOG["rfdetr-large"]["methods"],
                        value=MODEL_CATALOG["rfdetr-large"]["default_method"],
                        label="2. Model Method",
                        scale=1,
                    )
                    transport_dropdown = gr.Dropdown(
                        choices=[
                            ("WebSocket (/v1/ws) + HTTP Fallback", "ws_auto"),
                            ("WebSocket (/v1/ws) Only", "ws_only"),
                            ("HTTP (/chat/completions) Only", "http_only"),
                        ],
                        value="ws_auto",
                        label="3. Transport Protocol",
                        scale=2,
                    )
                with gr.Row():
                    prompt_textbox = gr.Textbox(
                        value="",
                        placeholder=MODEL_CATALOG["rfdetr-large"]["prompt_placeholder"],
                        label="Optional Prompt (Used by SAM 3.1 & Realtime VLMs)",
                        scale=4,
                    )
                    env_radio = gr.Radio(
                        choices=[("Dev Gateway (.env)", "dev"), ("Prod Gateway", "prod")],
                        value="dev",
                        label="Gateway Environment",
                        scale=2,
                    )

            with gr.Column(scale=3):
                model_card = gr.HTML(render_model_card_html("rfdetr-large"))

        # Advanced & Visualization Settings Accordion
        with gr.Accordion("Visualization & Stream Settings (Thresholds, Resolution, OCR Parameters)", open=False):
            with gr.Row():
                conf_slider = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    label="Confidence / Keypoint Threshold",
                )
                show_labels_chk = gr.Checkbox(
                    value=True,
                    label="Show Class / Score Badges on Video",
                )
                inflight_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=1,
                    label="WebSocket max_inflight (Concurrent Reads)",
                )
                max_dim_slider = gr.Slider(
                    minimum=384,
                    maximum=1280,
                    value=768,
                    step=64,
                    label="Max Frame Dimension (px)",
                )
                ocr_lang_box = gr.Textbox(
                    value="en",
                    label="PP-OCRv6 Language Code",
                    scale=1,
                )
                ocr_word_chk = gr.Checkbox(
                    value=False,
                    label="PP-OCRv6 Word Boxes",
                )

        # Main Tabs: Live Webcam Stream vs Snapshot / Image Test Bench
        with gr.Tabs():
            # ===============================================================
            # TAB 1: LIVE WEBCAM STREAMING
            # ===============================================================
            with gr.TabItem("🎥 Live Camera Stream"):
                telemetry_panel = gr.HTML(render_telemetry_html(LIVE_STREAMER.get_snapshot()))

                with gr.Row():
                    with gr.Column(scale=5):
                        webcam_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            label="Laptop Camera Feed (Click 'Record' / Camera Icon to Start Streaming)",
                            height=440,
                        )
                        stop_btn = gr.Button(
                            "⏹ Close Active WebSocket Session",
                            variant="stop",
                            size="sm",
                        )

                    with gr.Column(scale=5):
                        annotated_output = gr.Image(
                            type="numpy",
                            label="Live Gateway Detection / Segmentation / Pose Stream",
                            streaming=True,
                            height=440,
                        )

                    with gr.Column(scale=3):
                        live_summary_md = gr.Markdown(
                            value="### Live Output\n_Start your webcam stream to view real-time detections, OCR text, or VLM responses._"
                        )
                        with gr.Accordion("Raw Gateway Event / JSON Payload", open=False):
                            live_raw_json = gr.JSON(value={}, label="Latest Frame Event")

            # ===============================================================
            # TAB 2: SNAPSHOT & UPLOAD TEST BENCH
            # ===============================================================
            with gr.TabItem("📸 Snapshot / Image Test Bench"):
                gr.Markdown(
                    "Test any of the 8 VLM Run Gateway models on a single webcam snapshot, uploaded image, "
                    "or a generated demo card before starting continuous streaming."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        snap_input = gr.Image(
                            sources=["webcam", "upload", "clipboard"],
                            type="numpy",
                            label="Input Image or Webcam Snapshot",
                            height=420,
                        )
                        with gr.Row():
                            run_snap_btn = gr.Button("🚀 Run Gateway Inference", variant="primary")
                            demo_card_btn = gr.Button("🎨 Load Synthetic Test Card", variant="secondary")

                    with gr.Column(scale=5):
                        snap_output = gr.Image(
                            type="numpy",
                            label="Annotated Gateway Output",
                            height=420,
                        )

                    with gr.Column(scale=3):
                        snap_summary_md = gr.Markdown(value="### Single-Frame Output\n_Click **Run Gateway Inference** to inspect results._")
                        with gr.Accordion("Raw Gateway JSON Response", open=True):
                            snap_raw_json = gr.JSON(value={}, label="Gateway Response")

        # -------------------------------------------------------------------
        # Event Wiring
        # -------------------------------------------------------------------
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[method_dropdown, prompt_textbox, model_card],
        )

        env_radio.change(
            fn=render_header_html,
            inputs=[env_radio],
            outputs=[header_html],
        )

        stream_inputs = [
            webcam_input,
            model_dropdown,
            method_dropdown,
            prompt_textbox,
            transport_dropdown,
            env_radio,
            conf_slider,
            show_labels_chk,
            inflight_slider,
            max_dim_slider,
            ocr_lang_box,
            ocr_word_chk,
        ]
        stream_outputs = [
            annotated_output,
            telemetry_panel,
            live_summary_md,
            live_raw_json,
        ]

        webcam_input.stream(
            fn=process_webcam_stream,
            inputs=stream_inputs,
            outputs=stream_outputs,
            time_limit=600,
            stream_every=0.08,
            concurrency_limit=4,
        )

        stop_btn.click(
            fn=on_stop_stream,
            inputs=[],
            outputs=[telemetry_panel, live_summary_md],
        )

        demo_card_btn.click(
            fn=generate_demo_card_image,
            inputs=[],
            outputs=[snap_input],
        )

        run_snap_btn.click(
            fn=run_snapshot_inference,
            inputs=[
                snap_input,
                model_dropdown,
                method_dropdown,
                prompt_textbox,
                transport_dropdown,
                env_radio,
                conf_slider,
                show_labels_chk,
                max_dim_slider,
                ocr_lang_box,
                ocr_word_chk,
            ],
            outputs=[snap_output, snap_summary_md, snap_raw_json],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    launch_sig = inspect.signature(app.launch)
    launch_kwargs: dict[str, Any] = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "show_error": True,
    }
    if "css" in launch_sig.parameters:
        launch_kwargs["css"] = CUSTOM_CSS
    if "theme" in launch_sig.parameters:
        launch_kwargs["theme"] = gr.themes.Soft(
            primary_hue="sky",
            secondary_hue="purple",
            neutral_hue="slate",
            font=["Inter", "system-ui", "-apple-system", "sans-serif"],
            font_mono=["JetBrains Mono", "Consolas", "monospace"],
        )
    app.launch(**launch_kwargs)
