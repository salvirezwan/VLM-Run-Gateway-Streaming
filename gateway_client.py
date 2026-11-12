"""VLM Run Gateway WebSocket & HTTP Fallback Client for Live Video Streaming."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
import websockets
from dotenv import load_dotenv

from config import MODEL_CATALOG

# Load .env from project directory
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)


def _parse_env_keys() -> dict[str, str]:
    """Read active and commented dev/prod keys & URLs from .env for easy UI switching."""
    info = {
        "api_key": os.getenv("VLMRUN_API_KEY", "").strip(),
        "base_url": os.getenv("VLMRUN_BASE_URL", "https://dev-gateway.vlm.run/v1/openai").strip(),
        "dev_api_key": os.getenv("VLMRUN_API_KEY", "").strip(),
        "dev_base_url": "https://dev-gateway.vlm.run/v1/openai",
        "prod_api_key": "",
        "prod_base_url": "https://gateway.vlm.run/v1/openai",
    }
    if ENV_PATH.exists():
        text = ENV_PATH.read_text(encoding="utf-8", errors="ignore")
        m_prod = re.search(r"#\s*Prod\s*API\s*:\s*(\S+)", text, re.IGNORECASE)
        if m_prod:
            info["prod_api_key"] = m_prod.group(1).strip()
        m_dev = re.search(r"#\s*Dev\s*API\s*:\s*(\S+)", text, re.IGNORECASE)
        if m_dev:
            info["dev_api_key"] = m_dev.group(1).strip()
    return info


ENV_CONFIG = _parse_env_keys()


def derive_ws_url(base_url: str) -> str:
    """Convert https://dev-gateway.vlm.run/v1/openai -> wss://dev-gateway.vlm.run/v1/ws."""
    parsed = urlparse(base_url.strip())
    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    host = parsed.netloc or "dev-gateway.vlm.run"
    return f"{scheme}://{host}/v1/ws"


def derive_http_base_url(base_url: str) -> str:
    """Normalize HTTP OpenAI base URL (e.g., https://dev-gateway.vlm.run/v1/openai)."""
    url = base_url.strip().rstrip("/")
    if not url.endswith("/openai"):
        if url.endswith("/v1"):
            url = f"{url}/openai"
        else:
            url = f"{url}/v1/openai"
    return url


def encode_frame_jpeg(frame_rgb: np.ndarray, quality: int = 82, max_dim: int = 768) -> bytes:
    """Encode an RGB numpy frame into compact JPEG bytes (< 2 MiB limit)."""
    h, w = frame_rgb.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode video frame as JPEG")
    return buf.tobytes()


def parse_message_content(raw_content: Any) -> dict[str, Any] | str:
    """Parse `choices[0].message.content` into a structured dict or clean string."""
    if isinstance(raw_content, dict):
        return raw_content
    if not isinstance(raw_content, str):
        return str(raw_content)
    text = raw_content.strip()
    if not text:
        return ""
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


@dataclass
class StreamSessionConfig:
    """Configuration for an active live stream session."""

    model: str = "rfdetr-large"
    method: str = "detect"
    prompt: str = ""
    transport_mode: str = "ws_auto"  # "ws_auto", "ws_only", "http_only"
    environment: str = "dev"         # "dev" or "prod"
    max_inflight: int = 2
    jpeg_quality: int = 82
    max_dimension: int = 768
    ocr_lang: str = "en"
    ocr_word_boxes: bool = False

    def build_method_params(self) -> dict[str, Any] | None:
        params: dict[str, Any] = {}
        if self.model == "sam3.1" and self.method in ("segment", "track"):
            params["prompt"] = (self.prompt or "person").strip()
        elif self.model == "pp-ocrv6" and self.method == "ocr":
            if self.ocr_lang.strip():
                params["lang"] = self.ocr_lang.strip()
            if self.ocr_word_boxes:
                params["word_boxes"] = True
        return params if params else None

    def signature(self) -> tuple[Any, ...]:
        return (
            self.model,
            self.method,
            self.prompt.strip(),
            self.transport_mode,
            self.environment,
            self.max_inflight,
            self.ocr_lang.strip(),
            self.ocr_word_boxes,
        )


@dataclass
class TelemetryState:
    """Live telemetry and latest prediction state shared between worker and UI."""

    connected: bool = False
    active_transport: str = "Idle"
    session_id: str = "-"
    canonical_model: str = "-"
    max_fps: float = 20.0
    frames_sent: int = 0
    frames_read: int = 0
    frames_dropped: int = 0
    errors_count: int = 0
    last_latency_ms: float | None = None
    ema_fps: float = 0.0
    session_cost_usd: float = 0.0
    status_state: str = "ready"  # "ready", "streaming", "provisioning", "fallback", "error"
    status_message: str = "Ready to stream"
    latest_payload: dict[str, Any] | str | None = None
    latest_raw_response: dict[str, Any] = field(default_factory=dict)
    last_result_ts: float = 0.0


class GatewayLiveStreamer:
    """
    Manages a persistent WebSocket `/v1/ws` session with automatic HTTP `/v1/openai/chat/completions`
    fallback on a background asyncio event loop thread for zero-lag live video streaming.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.config = StreamSessionConfig()
        self.telemetry = TelemetryState()

        self._pending_frame_bytes: bytes | None = None
        self._frame_event: asyncio.Event | None = None
        self._stop_requested: bool = False
        self._reconnect_requested: bool = False
        self._last_frame_submit_ts: float = 0.0

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _ensure_worker_started(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._frame_event = asyncio.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="VLMRunGatewayWorker")
        self._thread.start()

    def _get_credentials(self, env_choice: str) -> tuple[str, str]:
        if env_choice == "prod" and ENV_CONFIG.get("prod_api_key"):
            return ENV_CONFIG["prod_api_key"], ENV_CONFIG["prod_base_url"]
        return ENV_CONFIG["dev_api_key"] or ENV_CONFIG["api_key"], ENV_CONFIG["base_url"]

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._worker_main())

    def update_config(self, new_cfg: StreamSessionConfig) -> None:
        """Update session parameters; triggers a clean session reconnect if changed."""
        with self._lock:
            if new_cfg.signature() != self.config.signature():
                model_changed = (new_cfg.model != self.config.model) or (new_cfg.method != self.config.method)
                self.config = new_cfg
                self._reconnect_requested = True
                if model_changed:
                    self.telemetry.latest_payload = None
                    self.telemetry.latest_raw_response = {}
                    self.telemetry.last_latency_ms = None
                if self._frame_event and self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._frame_event.set)

    def submit_frame(self, frame_rgb: np.ndarray) -> None:
        """Encode and queue the newest webcam frame (non-blocking, drops stale unsent frame)."""
        self._ensure_worker_started()
        with self._lock:
            cfg = self.config
        try:
            jpg_bytes = encode_frame_jpeg(
                frame_rgb,
                quality=cfg.jpeg_quality,
                max_dim=cfg.max_dimension,
            )
        except Exception:
            return

        with self._lock:
            self._pending_frame_bytes = jpg_bytes
            self._last_frame_submit_ts = time.time()
            self._stop_requested = False

        if self._frame_event and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._frame_event.set)

    def stop_session(self) -> None:
        """Explicitly close any open WebSocket session so idle time is not billed."""
        with self._lock:
            self._stop_requested = True
            self._pending_frame_bytes = None
        if self._frame_event and self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._frame_event.set)

    def get_snapshot(self) -> TelemetryState:
        """Return a thread-safe shallow copy of current telemetry & latest prediction."""
        with self._lock:
            return TelemetryState(
                connected=self.telemetry.connected,
                active_transport=self.telemetry.active_transport,
                session_id=self.telemetry.session_id,
                canonical_model=self.telemetry.canonical_model,
                max_fps=self.telemetry.max_fps,
                frames_sent=self.telemetry.frames_sent,
                frames_read=self.telemetry.frames_read,
                frames_dropped=self.telemetry.frames_dropped,
                errors_count=self.telemetry.errors_count,
                last_latency_ms=self.telemetry.last_latency_ms,
                ema_fps=self.telemetry.ema_fps,
                session_cost_usd=self.telemetry.session_cost_usd,
                status_state=self.telemetry.status_state,
                status_message=self.telemetry.status_message,
                latest_payload=self.telemetry.latest_payload,
                latest_raw_response=dict(self.telemetry.latest_raw_response),
                last_result_ts=self.telemetry.last_result_ts,
            )

    async def _worker_main(self) -> None:
        assert self._frame_event is not None
        while True:
            await self._frame_event.wait()
            self._frame_event.clear()

            with self._lock:
                if self._stop_requested or self._pending_frame_bytes is None:
                    continue
                cfg = self.config
                self._reconnect_requested = False

            if cfg.transport_mode == "http_only":
                await self._run_http_loop(cfg)
            else:
                await self._run_ws_or_fallback_loop(cfg)

    async def _run_ws_or_fallback_loop(self, cfg: StreamSessionConfig) -> None:
        """Connect to WS /v1/ws, stream frames, and fall back to HTTP if provisioning or unavailable."""
        api_key, base_url = self._get_credentials(cfg.environment)
        ws_url = derive_ws_url(base_url)

        with self._lock:
            self.telemetry.status_state = "streaming"
            self.telemetry.status_message = f"Connecting to {ws_url} ({cfg.model})..."
            self.telemetry.active_transport = "WS Connecting"

        create_msg: dict[str, Any] = {
            "type": "session.create",
            "model": cfg.model,
            "method": cfg.method,
            "max_inflight": cfg.max_inflight,
        }
        method_params = cfg.build_method_params()
        if method_params:
            create_msg["method_params"] = method_params
        if cfg.prompt.strip() and cfg.model in ("gemma-4-e2b-it-realtime", "qwen3.5-0.8b-realtime"):
            create_msg["prompt"] = cfg.prompt.strip()

        try:
            async with websockets.connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                open_timeout=12,
                ping_interval=15,
                max_size=8 * 1024 * 1024,
            ) as ws:
                await ws.send(json.dumps(create_msg))
                raw_first = await asyncio.wait_for(ws.recv(), timeout=15)
                first_ev = json.loads(raw_first)

                if first_ev.get("type") == "error":
                    err_type = first_ev.get("error_type", "error")
                    err_msg = first_ev.get("message", "Session creation refused")
                    retry_after = first_ev.get("retry_after")
                    with self._lock:
                        self.telemetry.connected = False
                        self.telemetry.errors_count += 1
                        if err_type == "model_provisioning_error":
                            self.telemetry.status_state = "provisioning"
                            self.telemetry.status_message = (
                                f"Replica starting up: {err_msg}"
                                + (" — Using HTTP fallback meanwhile" if cfg.transport_mode == "ws_auto" else "")
                            )
                        else:
                            self.telemetry.status_state = "error"
                            self.telemetry.status_message = f"WS {err_type}: {err_msg}"

                    if cfg.transport_mode == "ws_auto":
                        await self._run_http_loop(cfg, retry_ws_after=min(float(retry_after or 25.0), 30.0))
                    else:
                        await asyncio.sleep(2.0)
                    return

                if first_ev.get("type") != "session.created":
                    with self._lock:
                        self.telemetry.status_state = "error"
                        self.telemetry.status_message = f"Unexpected WS event: {first_ev}"
                    return

                max_fps = float(first_ev.get("max_fps") or 20.0)
                min_interval = 1.0 / max(1.0, max_fps)

                with self._lock:
                    self.telemetry.connected = True
                    self.telemetry.active_transport = "WS /v1/ws"
                    self.telemetry.session_id = first_ev.get("session_id", "-")
                    self.telemetry.canonical_model = first_ev.get("model", cfg.model)
                    self.telemetry.max_fps = max_fps
                    self.telemetry.status_state = "streaming"
                    self.telemetry.status_message = (
                        f"Live WebSocket session {first_ev.get('session_id')} active "
                        f"(max {max_fps:.0f} FPS, max_inflight={first_ev.get('max_inflight')})"
                    )

                receiver_task = asyncio.create_task(self._ws_receiver(ws))
                last_send_ts = 0.0

                try:
                    while not receiver_task.done():
                        with self._lock:
                            stop_now = self._stop_requested or self._reconnect_requested
                            idle_for = time.time() - self._last_frame_submit_ts
                            frame_bytes = self._pending_frame_bytes
                            self._pending_frame_bytes = None

                        if stop_now or idle_for > 12.0:
                            # Cleanly close session so idle time is not billed
                            try:
                                await ws.send(json.dumps({"type": "session.close"}))
                                await asyncio.wait_for(receiver_task, timeout=2.5)
                            except Exception:
                                pass
                            break

                        if frame_bytes is not None:
                            now = time.time()
                            wait_s = min_interval - (now - last_send_ts)
                            if wait_s > 0:
                                await asyncio.sleep(wait_s)
                            await ws.send(frame_bytes)
                            last_send_ts = time.time()
                            with self._lock:
                                self.telemetry.frames_sent += 1
                        else:
                            await asyncio.sleep(0.015)
                finally:
                    if not receiver_task.done():
                        receiver_task.cancel()
                    with self._lock:
                        self.telemetry.connected = False
                        if self._stop_requested:
                            self.telemetry.active_transport = "Stopped"
                            self.telemetry.status_state = "ready"
                            self.telemetry.status_message = "Stream stopped (WebSocket session closed cleanly)"

        except Exception as exc:
            with self._lock:
                self.telemetry.connected = False
                self.telemetry.errors_count += 1
                self.telemetry.status_state = "fallback" if cfg.transport_mode == "ws_auto" else "error"
                self.telemetry.status_message = f"WS connection notice: {exc}"
            if cfg.transport_mode == "ws_auto":
                await self._run_http_loop(cfg, retry_ws_after=20.0)
            else:
                await asyncio.sleep(1.5)

    async def _ws_receiver(self, ws: Any) -> None:
        """Receive `result`, `frame.dropped`, `error`, and `session.stats` events from WS /v1/ws."""
        async for raw_msg in ws:
            if not isinstance(raw_msg, str):
                continue
            event = json.loads(raw_msg)
            ev_type = event.get("type")

            if ev_type == "result":
                lat = float(event.get("latency_ms") or 0.0)
                resp = event.get("response", {})
                choices = resp.get("choices", [])
                raw_content = choices[0].get("message", {}).get("content", "") if choices else ""
                parsed = parse_message_content(raw_content)
                now = time.time()
                with self._lock:
                    if self.telemetry.last_result_ts > 0:
                        inst_fps = 1.0 / max(0.001, now - self.telemetry.last_result_ts)
                        self.telemetry.ema_fps = (
                            0.75 * self.telemetry.ema_fps + 0.25 * inst_fps
                            if self.telemetry.ema_fps > 0
                            else inst_fps
                        )
                    self.telemetry.last_result_ts = now
                    self.telemetry.frames_read += 1
                    self.telemetry.last_latency_ms = lat
                    self.telemetry.latest_payload = parsed
                    self.telemetry.latest_raw_response = event

            elif ev_type == "frame.dropped":
                with self._lock:
                    self.telemetry.frames_dropped += 1

            elif ev_type == "error":
                with self._lock:
                    self.telemetry.errors_count += 1
                    self.telemetry.status_message = (
                        f"Frame {event.get('frame')} error ({event.get('error_type')}): {event.get('message')}"
                    )

            elif ev_type == "session.stats":
                with self._lock:
                    cost = event.get("cost")
                    if cost is not None:
                        self.telemetry.session_cost_usd += float(cost)
                    self.telemetry.latest_raw_response = event
                break

    async def _run_http_loop(self, cfg: StreamSessionConfig, retry_ws_after: float | None = None) -> None:
        """Send frames via HTTP POST /v1/openai/chat/completions (fallback or explicit HTTP mode)."""
        api_key, base_url = self._get_credentials(cfg.environment)
        http_base = derive_http_base_url(base_url)
        endpoint = f"{http_base}/chat/completions"
        started_http_at = time.time()

        with self._lock:
            self.telemetry.active_transport = "HTTP /chat/completions"
            if self.telemetry.status_state != "provisioning":
                self.telemetry.status_state = "streaming"
                self.telemetry.status_message = f"Streaming via HTTP POST ({endpoint})"

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                with self._lock:
                    if self._stop_requested or self._reconnect_requested:
                        break
                    if retry_ws_after and (time.time() - started_http_at) >= retry_ws_after:
                        # Periodically retry WebSocket if we are in ws_auto mode
                        break
                    if (time.time() - self._last_frame_submit_ts) > 12.0:
                        break
                    frame_bytes = self._pending_frame_bytes
                    self._pending_frame_bytes = None

                if frame_bytes is None:
                    await asyncio.sleep(0.02)
                    continue

                t0 = time.perf_counter()
                with self._lock:
                    self.telemetry.frames_sent += 1

                res = await self._send_single_http_frame(client, endpoint, api_key, cfg, frame_bytes)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if res["ok"]:
                    now = time.time()
                    with self._lock:
                        if self.telemetry.last_result_ts > 0:
                            inst_fps = 1.0 / max(0.001, now - self.telemetry.last_result_ts)
                            self.telemetry.ema_fps = (
                                0.75 * self.telemetry.ema_fps + 0.25 * inst_fps
                                if self.telemetry.ema_fps > 0
                                else inst_fps
                            )
                        self.telemetry.last_result_ts = now
                        self.telemetry.frames_read += 1
                        self.telemetry.last_latency_ms = elapsed_ms
                        self.telemetry.canonical_model = res.get("model", cfg.model)
                        self.telemetry.latest_payload = res["parsed"]
                        self.telemetry.latest_raw_response = res["raw"]
                        self.telemetry.status_state = "streaming"
                        self.telemetry.status_message = f"HTTP active ({elapsed_ms:.0f} ms)"
                else:
                    with self._lock:
                        self.telemetry.errors_count += 1
                        self.telemetry.status_message = res["error"]
                    await asyncio.sleep(1.0)

    async def _send_single_http_frame(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        api_key: str,
        cfg: StreamSessionConfig,
        frame_bytes: bytes,
    ) -> dict[str, Any]:
        b64_img = base64.b64encode(frame_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64_img}"

        # Map realtime-only aliases when falling back to HTTP chat completions
        http_model = cfg.model
        if cfg.model == "qwen3.5-0.8b-realtime":
            http_model = "qwen/qwen3.5-0.8b"
        elif cfg.model == "gemma-4-e2b-it-realtime":
            http_model = "google/gemma4-26b-a4b-it"

        content_parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_uri}}
        ]
        if cfg.prompt.strip() and (
            cfg.model in ("gemma-4-e2b-it-realtime", "qwen3.5-0.8b-realtime")
            or (cfg.model == "sam3.1" and cfg.method in ("segment", "track"))
        ):
            content_parts.append({"type": "text", "text": cfg.prompt.strip()})

        body: dict[str, Any] = {
            "model": http_model,
            "messages": [{"role": "user", "content": content_parts}],
        }
        if cfg.method:
            body["method"] = cfg.method
        method_params = cfg.build_method_params()
        if method_params:
            body["method_params"] = method_params

        # Use json_object response format for structured vision models
        if cfg.model not in ("gemma-4-e2b-it-realtime", "qwen3.5-0.8b-realtime") and cfg.method != "text":
            body["response_format"] = {"type": "json_object"}

        try:
            resp = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            if resp.status_code != 200:
                return {
                    "ok": False,
                    "error": f"HTTP {resp.status_code}: {resp.text[:240]}",
                }
            data = resp.json()
            choices = data.get("choices", [])
            raw_content = choices[0].get("message", {}).get("content", "") if choices else ""
            parsed = parse_message_content(raw_content)
            return {
                "ok": True,
                "model": data.get("served_model_id") or data.get("model") or http_model,
                "parsed": parsed,
                "raw": data,
            }
        except Exception as exc:
            return {"ok": False, "error": f"HTTP request error: {exc}"}


async def run_single_frame_test(
    frame_rgb: np.ndarray,
    cfg: StreamSessionConfig,
) -> tuple[dict[str, Any] | str | None, dict[str, Any], float, str]:
    """
    Run a single-frame test over WebSocket (or HTTP fallback) and return:
    (parsed_payload, raw_response, latency_ms, status_text)
    """
    api_key, base_url = (
        (ENV_CONFIG["prod_api_key"], ENV_CONFIG["prod_base_url"])
        if cfg.environment == "prod" and ENV_CONFIG.get("prod_api_key")
        else (ENV_CONFIG["dev_api_key"] or ENV_CONFIG["api_key"], ENV_CONFIG["base_url"])
    )
    jpg_bytes = encode_frame_jpeg(frame_rgb, quality=cfg.jpeg_quality, max_dim=cfg.max_dimension)

    if cfg.transport_mode != "http_only":
        ws_url = derive_ws_url(base_url)
        create_msg: dict[str, Any] = {
            "type": "session.create",
            "model": cfg.model,
            "method": cfg.method,
            "max_inflight": 1,
        }
        mp = cfg.build_method_params()
        if mp:
            create_msg["method_params"] = mp
        if cfg.prompt.strip() and cfg.model in ("gemma-4-e2b-it-realtime", "qwen3.5-0.8b-realtime"):
            create_msg["prompt"] = cfg.prompt.strip()

        try:
            t0 = time.perf_counter()
            async with websockets.connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                open_timeout=10,
                max_size=8 * 1024 * 1024,
            ) as ws:
                await ws.send(json.dumps(create_msg))
                created = json.loads(await asyncio.wait_for(ws.recv(), timeout=12))
                if created.get("type") == "session.created":
                    await ws.send(jpg_bytes)
                    ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=25))
                    await ws.send(json.dumps({"type": "session.close"}))
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except Exception:
                        pass
                    if ev.get("type") == "result":
                        lat = float(ev.get("latency_ms") or ((time.perf_counter() - t0) * 1000.0))
                        choices = ev.get("response", {}).get("choices", [])
                        raw_c = choices[0].get("message", {}).get("content", "") if choices else ""
                        return (
                            parse_message_content(raw_c),
                            ev,
                            lat,
                            f"WebSocket /v1/ws success ({lat:.0f} ms)",
                        )
                    return None, ev, 0.0, f"WS event: {ev.get('message') or ev.get('type')}"
                elif cfg.transport_mode == "ws_only":
                    return None, created, 0.0, f"WS {created.get('error_type')}: {created.get('message')}"
        except Exception as exc:
            if cfg.transport_mode == "ws_only":
                return None, {"error": str(exc)}, 0.0, f"WS Error: {exc}"

    # HTTP Fallback or HTTP-only mode
    http_base = derive_http_base_url(base_url)
    endpoint = f"{http_base}/chat/completions"
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=35.0) as client:
        dummy_streamer = GatewayLiveStreamer.__new__(GatewayLiveStreamer)
        res = await dummy_streamer._send_single_http_frame(client, endpoint, api_key, cfg, jpg_bytes)
        lat = (time.perf_counter() - t0) * 1000.0
        if res["ok"]:
            return res["parsed"], res["raw"], lat, f"HTTP /chat/completions success ({lat:.0f} ms)"
        return None, {"error": res["error"]}, lat, res["error"]
