# VLM Run Gateway — Live Video Detection App

A Gradio web app for real-time computer vision and multimodal AI on live webcam streams using the **VLM Run Gateway** (`WS /v1/ws` WebSocket with automatic HTTP fallback).

## Supported Models

- **`rfdetr-large`** — Real-time COCO object detection (bounding boxes)
- **`rfdetr-seg-large`** — Real-time COCO instance segmentation (pixel masks + boxes)
- **`sam3.1`** — Zero-shot promptable object segmentation and tracking
- **`vitpose-plus-large`** — 2D human body pose estimation (17 COCO keypoints)
- **`hamer`** — 2D & 3D hand pose and mesh recovery (21 hand joints)
- **`pp-ocrv6`** — Scene and document text detection & OCR
- **`gemma-4-e2b-it-realtime`** — Live multimodal VLM chat & captioning
- **`qwen3.5-0.8b-realtime`** — Ultra-fast live multimodal VLM chat

---

## How to Download and Run

### 1. Clone the repository

```bash
git clone https://github.com/salvirezwan/VLM-Run-Gateway-Streaming.git
cd VLM-Run-Gateway-Streaming
```

### 2. Create a virtual environment and install dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project folder (or copy `.env.example` to `.env`) and add your VLM Run credentials:

```env
VLMRUN_API_KEY=your_api_key_here
VLMRUN_BASE_URL=https://gateway.vlm.run/v1/openai
```

### 4. Run the app

```bash
python app.py
```

Open **`http://127.0.0.1:7860`** in your browser, allow camera access, pick a model from the dropdown, and click the **Record / Start** button on the webcam feed to start streaming.
