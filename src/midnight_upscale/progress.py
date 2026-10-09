"""Live progress for a Midnight Upscale job.

ComfyUI WebSocket messages are reduced to one event. The GUI and the CLI both
read that event. Sampler steps are never reported as finished frames.
"""

from __future__ import annotations

import json
import struct
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from midnight_upscale.utils import PipelineError

STAGES: tuple[str, ...] = (
    "Inspect input",
    "Decode GIF",
    "Prepare RGB / alpha",
    "Upscale alpha",
    "Build SeedVR2 input",
    "Queue ComfyUI job",
    "SeedVR2 processing",
    "Retrieve SeedVR2 result",
    "Decode upscaled RGB",
    "Recombine RGBA",
    "Encode output",
    "Validate output",
    "Finished",
)

_SEED_STAGE = "SeedVR2 processing"
# A finished upscaled frame exists only after SeedVR2 has written RGB or RGBA.
COMPLETED_PREVIEW_STAGES = frozenset({"Decode upscaled RGB", "Recombine RGBA"})

PREVIEW_IMAGE = 1
PREVIEW_IMAGE_WITH_METADATA = 4

TRACKED_EVENTS = frozenset(
    {
        "execution_start",
        "executing",
        "progress",
        "progress_state",
        "executed",
        "execution_success",
        "execution_error",
        "execution_interrupted",
    }
)

_bus: ContextVar[ProgressBus | None] = ContextVar("midnight_progress", default=None)


class JobCancelled(PipelineError):
    """The user cancelled this prompt. Partial output is not a success."""


def current_bus() -> ProgressBus | None:
    return _bus.get()


def bind_bus(bus: ProgressBus) -> Token[ProgressBus | None]:
    return _bus.set(bus)


def reset_bus(token: Token[ProgressBus | None]) -> None:
    _bus.reset(token)


def checkpoint() -> None:
    bus = current_bus()
    if bus is not None and bus.cancel_requested():
        bus.mark_cancelled("Cancelled before the next stage.")
        raise JobCancelled("Cancelled before the next stage.")


def report(
    stage: str,
    *,
    message: str = "",
    frames_done: int | None = None,
    frames_total: int | None = None,
    frames_kind: str = "",
    batch_index: int | None = None,
    batch_total: int | None = None,
    preview_path: str = "",
) -> None:
    bus = current_bus()
    if bus is None:
        return
    bus.update_stage(
        stage,
        message=message,
        frames_done=frames_done,
        frames_total=frames_total,
        frames_kind=frames_kind,
        batch_index=batch_index,
        batch_total=batch_total,
        preview_path=preview_path,
    )


@dataclass
class PipelineProgressEvent:
    stage: str = STAGES[0]
    stage_index: int = 1
    stage_count: int = len(STAGES)
    status: str = "ACTIVE"
    frames_done: int | None = None
    frames_total: int | None = None
    frames_kind: str = ""
    batch_index: int | None = None
    batch_total: int | None = None
    comfy_prompt_id: str = ""
    comfy_node_id: str = ""
    comfy_node_type: str = ""
    node_step: int | None = None
    node_steps_total: int | None = None
    elapsed_seconds: float = 0.0
    last_event_age_seconds: float | None = None
    message: str = ""
    preview_bytes: bytes | None = None
    preview_path: str = ""
    preview_kind: str = ""
    overall_percent: float | None = None
    queue_state: str = ""
    error_node: str = ""
    error_type: str = ""
    error_message: str = ""
    eta_seconds: float | None = None
    gpu_name: str = ""
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    ws_alive: bool = False
    health: str = ""


Listener = Callable[[PipelineProgressEvent], None]


@dataclass
class ProgressBus:
    """One job. WebSocket messages for other prompt ids are ignored."""

    started: float = field(default_factory=time.monotonic)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    stage: str = STAGES[0]
    status: str = "ACTIVE"
    frames_done: int | None = None
    frames_total: int | None = None
    frames_kind: str = ""
    batch_index: int | None = None
    batch_total: int | None = None
    prompt_id: str = ""
    node_id: str = ""
    node_type: str = ""
    node_step: int | None = None
    node_steps_total: int | None = None
    nodes_done: int = 0
    node_order: list[str] = field(default_factory=list)
    node_types: dict[str, str] = field(default_factory=dict)
    message: str = ""
    preview_bytes: bytes | None = None
    preview_path: str = ""
    preview_kind: str = ""
    queue_state: str = ""
    error_node: str = ""
    error_type: str = ""
    error_message: str = ""
    gpu_name: str = ""
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    ws_alive: bool = False
    reconnects: int = 0
    health: str = ""
    comfy_failed: str = ""
    last_comfy_at: float | None = None
    last_step_at: float | None = None
    last_step_value: int | None = None
    _listeners: list[Listener] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _socket_was_down: bool = False

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    def assign_prompt(self, prompt_id: str) -> None:
        with self._lock:
            self.prompt_id = prompt_id
            self.queue_state = "WAITING"
            self.status = "WAITING"
            self.stage = "Queue ComfyUI job"
            self.frames_done = None
            self.frames_kind = "submitted"
            self.message = f"ComfyUI prompt queued: {prompt_id[:8]}"
        self._publish(self.message)

    def set_nodes(self, node_types: dict[str, str]) -> None:
        with self._lock:
            self.node_types = dict(node_types)
            self.node_order = list(node_types)
        self._publish("")

    def update_stage(
        self,
        stage: str,
        *,
        message: str = "",
        frames_done: int | None = None,
        frames_total: int | None = None,
        frames_kind: str = "",
        batch_index: int | None = None,
        batch_total: int | None = None,
        preview_path: str = "",
    ) -> None:
        checkpoint_cancel = self.cancel_requested()
        with self._lock:
            self.stage = stage
            if stage != _SEED_STAGE:
                self.node_id = ""
                self.node_type = ""
                self.node_step = None
                self.node_steps_total = None
            if frames_done is not None:
                self.frames_done = frames_done
            if frames_total is not None:
                self.frames_total = frames_total
            if frames_kind:
                self.frames_kind = frames_kind
            if stage == _SEED_STAGE:
                # A sampler step is not a finished output frame.
                self.frames_done = None
                self.frames_kind = "submitted"
                if self.preview_kind == "completed":
                    self.preview_path = ""
                    self.preview_kind = ""
            if batch_index is not None:
                self.batch_index = batch_index
            if batch_total is not None:
                self.batch_total = batch_total
            if preview_path and stage in COMPLETED_PREVIEW_STAGES:
                self.preview_path = preview_path
                self.preview_kind = "completed"
                self.preview_bytes = None
            if message:
                self.message = message
            if checkpoint_cancel:
                self.status = "CANCELLED"
            elif self.status not in {"FAILED", "CANCELLED", "FINISHED"}:
                self.status = "ACTIVE"
        self._publish(message)

    def note_queue(
        self,
        queue_state: str,
        *,
        ws_alive: bool,
        history_present: bool = False,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            self.queue_state = queue_state
            self.ws_alive = ws_alive
            age = None if self.last_comfy_at is None else now - self.last_comfy_at
            quiet = age is not None and age >= 60
            if self.status in {"FAILED", "CANCELLED", "FINISHED"}:
                pass
            elif queue_state == "WAITING":
                self.status = "WAITING"
                self.health = "The prompt is still waiting in the ComfyUI queue."
            elif queue_state == "UNKNOWN" and not history_present and self.prompt_id:
                self.status = "UNKNOWN"
                self.health = (
                    "The prompt left the queue and history has no success or error for it."
                )
            elif quiet and queue_state == "RUNNING":
                self.status = "ACTIVE — no recent progress event"
                alive = "alive" if ws_alive else "reconnecting"
                self.health = (
                    f"No progress event for {age:.0f} seconds. "
                    f"ComfyUI connection is {alive}. "
                    "Job is still present in the running queue."
                )
            elif not ws_alive and self.prompt_id:
                self.health = (
                    "WebSocket disconnected. Reconnecting. The prompt was not submitted again."
                )
            else:
                self.status = "ACTIVE"
                self.health = ""
        self._publish("")

    def note_gpu(self, name: str, used_mb: float | None, total_mb: float | None) -> None:
        with self._lock:
            self.gpu_name = name
            self.vram_used_mb = used_mb
            self.vram_total_mb = total_mb
        self._publish("")

    def note_socket(self, alive: bool) -> None:
        with self._lock:
            if alive and self._socket_was_down:
                self.reconnects += 1
            self._socket_was_down = not alive
            self.ws_alive = alive
        self._publish("")

    def mark_cancelled(self, message: str) -> None:
        with self._lock:
            self.status = "CANCELLED"
            self.message = message
        self._publish(message)

    def mark_finished(self, message: str) -> None:
        with self._lock:
            self.stage = "Finished"
            self.status = "FINISHED"
            self.message = message
        self._publish(message)

    def apply_socket(self, message: str | bytes) -> None:
        if isinstance(message, (bytes, bytearray)):
            self._apply_binary(bytes(message))
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        event = str(payload.get("type") or "")
        if event not in TRACKED_EVENTS:
            return
        data = payload.get("data")
        if not isinstance(data, dict):
            return
        if not self._matches_prompt(data, event):
            return
        self._apply_event(event, data)

    def snapshot(self) -> PipelineProgressEvent:
        now = time.monotonic()
        with self._lock:
            age = None if self.last_comfy_at is None else max(0.0, now - self.last_comfy_at)
            stage_index = STAGES.index(self.stage) + 1 if self.stage in STAGES else 1
            return PipelineProgressEvent(
                stage=self.stage,
                stage_index=stage_index,
                stage_count=len(STAGES),
                status=self.status,
                frames_done=self.frames_done,
                frames_total=self.frames_total,
                frames_kind=self.frames_kind,
                batch_index=self.batch_index,
                batch_total=self.batch_total,
                comfy_prompt_id=self.prompt_id,
                comfy_node_id=self.node_id,
                comfy_node_type=self.node_type,
                node_step=self.node_step,
                node_steps_total=self.node_steps_total,
                elapsed_seconds=max(0.0, now - self.started),
                last_event_age_seconds=age,
                message=self.message,
                preview_bytes=self.preview_bytes,
                preview_path=self.preview_path,
                preview_kind=self.preview_kind,
                overall_percent=None,
                queue_state=self.queue_state,
                error_node=self.error_node,
                error_type=self.error_type,
                error_message=self.error_message,
                eta_seconds=self._eta_locked(now),
                gpu_name=self.gpu_name,
                vram_used_mb=self.vram_used_mb,
                vram_total_mb=self.vram_total_mb,
                ws_alive=self.ws_alive,
                health=self.health,
            )

    def _matches_prompt(self, data: dict[str, Any], event: str) -> bool:
        incoming = str(data.get("prompt_id") or "")
        if event == "progress_state" and not incoming:
            nodes = data.get("nodes")
            if isinstance(nodes, dict):
                for state in nodes.values():
                    if isinstance(state, dict) and state.get("prompt_id"):
                        incoming = str(state["prompt_id"])
                        break
        with self._lock:
            if not self.prompt_id or not incoming:
                return False
            return incoming == self.prompt_id

    def _apply_event(self, event: str, data: dict[str, Any]) -> None:
        message = ""
        with self._lock:
            self.last_comfy_at = time.monotonic()
            if self.status in {"ACTIVE — no recent progress event", "WAITING", "UNKNOWN"}:
                self.status = "ACTIVE"
                self.health = ""
            if event == "execution_start":
                self.stage = _SEED_STAGE
                self.frames_done = None
                self.frames_kind = "submitted"
                self.queue_state = "RUNNING"
                message = f"ComfyUI prompt queued: {self.prompt_id[:8]}"
            elif event == "executing":
                node = data.get("node")
                if node:
                    self.stage = _SEED_STAGE
                    self.node_id = str(node)
                    self.node_type = self.node_types.get(self.node_id, "")
                    self.node_step = None
                    self.node_steps_total = None
                    if self.node_id in self.node_order:
                        self.nodes_done = self.node_order.index(self.node_id)
                    label = self.node_type or self.node_id
                    message = f"Executing {label}"
            elif event == "progress":
                value = _as_int(data.get("value"))
                maximum = _as_int(data.get("max"))
                node = data.get("node")
                if node:
                    self.node_id = str(node)
                    self.node_type = self.node_types.get(self.node_id, self.node_type)
                if value is not None:
                    self.node_step = value
                if maximum is not None:
                    self.node_steps_total = maximum
                self._remember_step_locked(value)
                if self.node_step is not None and self.node_steps_total:
                    label = self.node_type or self.node_id or "node"
                    message = f"{label} {self.node_step}/{self.node_steps_total}"
            elif event == "progress_state":
                message = self._apply_progress_state_locked(data)
            elif event == "executed":
                node = str(data.get("node") or "")
                if node and node in self.node_order:
                    self.nodes_done = self.node_order.index(node) + 1
            elif event == "execution_success":
                self.queue_state = ""
                message = "ComfyUI prompt finished"
            elif event == "execution_error":
                self.status = "FAILED"
                self.error_node = str(data.get("node_id") or data.get("node") or "")
                self.error_type = str(data.get("node_type") or data.get("exception_type") or "")
                self.error_message = str(data.get("exception_message") or "ComfyUI execution error")
                self.node_id = self.error_node
                self.node_type = str(data.get("node_type") or self.node_type)
                self.comfy_failed = self.error_message
                message = self.error_message
            elif event == "execution_interrupted":
                self.status = "CANCELLED"
                self.error_node = str(data.get("node_id") or "")
                self.error_type = str(data.get("node_type") or "")
                self.error_message = "ComfyUI interrupted this prompt."
                message = self.error_message
            if message:
                self.message = message
        self._publish(message)

    def _apply_progress_state_locked(self, data: dict[str, Any]) -> str:
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            return ""
        running: tuple[str, dict[str, Any]] | None = None
        for node_id, state in nodes.items():
            if not isinstance(state, dict):
                continue
            if str(state.get("prompt_id") or self.prompt_id) != self.prompt_id:
                continue
            if str(state.get("state") or "") == "running":
                running = (str(node_id), state)
        if running is None:
            return ""
        node_id, state = running
        self.node_id = node_id
        self.node_type = self.node_types.get(node_id, self.node_type)
        value = _as_int(state.get("value"))
        maximum = _as_int(state.get("max"))
        if value is not None:
            self.node_step = value
        if maximum is not None:
            self.node_steps_total = maximum
        self._remember_step_locked(value)
        if self.node_step is None or not self.node_steps_total:
            return f"Executing {self.node_type or node_id}"
        label = self.node_type or node_id
        return f"{label} {self.node_step}/{self.node_steps_total}"

    def _remember_step_locked(self, value: int | None) -> None:
        if value is None:
            return
        if self.last_step_value != value:
            self.last_step_value = value
            self.last_step_at = time.monotonic()

    def _eta_locked(self, now: float) -> float | None:
        if (
            self.node_step is None
            or not self.node_steps_total
            or self.node_step < 3
            or self.last_step_at is None
        ):
            return None
        elapsed = now - self.started
        if elapsed < 15 or self.node_step <= 0:
            return None
        rate = elapsed / self.node_step
        remaining = self.node_steps_total - self.node_step
        if remaining <= 0:
            return None
        return rate * remaining

    def _apply_binary(self, blob: bytes) -> None:
        preview = parse_preview_bytes(blob, self.prompt_id)
        if preview is None:
            return
        with self._lock:
            if not self.prompt_id:
                return
            self.preview_bytes = preview
            self.preview_kind = "comfy"
            self.preview_path = ""
            self.last_comfy_at = time.monotonic()
            self.message = "ComfyUI AI preview — not final output"
        self._publish(self.message)

    def _publish(self, message: str) -> None:
        event = self.snapshot()
        if message:
            event.message = message
        for listener in list(self._listeners):
            listener(event)


def parse_preview_bytes(blob: bytes, prompt_id: str) -> bytes | None:
    """Return image bytes only when they belong to this prompt, or are unlabeled on our socket."""

    if len(blob) < 8:
        return None
    event = struct.unpack_from(">I", blob, 0)[0]
    body = blob[4:]
    if event == PREVIEW_IMAGE_WITH_METADATA:
        if len(body) < 4:
            return None
        meta_len = struct.unpack_from(">I", body, 0)[0]
        if meta_len < 0 or 4 + meta_len > len(body):
            return None
        try:
            meta = json.loads(body[4 : 4 + meta_len])
        except json.JSONDecodeError:
            return None
        if not isinstance(meta, dict) or str(meta.get("prompt_id") or "") != prompt_id:
            return None
        image = bytes(body[4 + meta_len :])
        return image or None
    if event == PREVIEW_IMAGE:
        # This frame has no prompt id. Accept it only after our prompt is known,
        # because the socket itself is bound to our client id.
        if not prompt_id or len(body) <= 4:
            return None
        image = bytes(body[4:])
        return image or None
    return None


def format_status(event: PipelineProgressEvent) -> str:
    lines = ["STATUS", event.status, "", "Pipeline stage:", event.stage]
    if event.comfy_prompt_id:
        lines.extend(["", "ComfyUI job:", event.comfy_prompt_id[:8]])
    if event.queue_state:
        lines.extend(["", "Queue:", event.queue_state])
    node = event.comfy_node_type or event.comfy_node_id
    if node:
        lines.extend(["", "Current node:", node])
        if event.node_step is not None and event.node_steps_total:
            lines.extend(
                [
                    "",
                    "Node progress:",
                    f"{event.node_step} / {event.node_steps_total}",
                    _bar(100.0 * event.node_step / event.node_steps_total),
                ]
            )
        else:
            lines.extend(["", "Node progress:", "Not reported by ComfyUI", f"Processing {node}..."])
    if event.batch_index is not None and event.batch_total:
        lines.extend(["", "Chunk:", f"{event.batch_index} / {event.batch_total}"])
    if event.frames_total and event.stage == _SEED_STAGE:
        lines.extend(["", "Frames submitted:", str(event.frames_total)])
    elif event.frames_total and event.frames_kind == "completed" and event.frames_done is not None:
        lines.extend(["", "Frames completed:", f"{event.frames_done} / {event.frames_total}"])
    elif event.frames_total and event.frames_kind == "submitted":
        lines.extend(["", "Frames submitted:", str(event.frames_total)])
    lines.extend(["", "Elapsed:", _clock(event.elapsed_seconds)])
    if event.comfy_prompt_id:
        if event.last_event_age_seconds is None:
            lines.extend(["", "Last WebSocket event:", "none yet"])
        else:
            lines.extend(
                ["", "Last WebSocket event:", f"{event.last_event_age_seconds:.1f} sec ago"]
            )
    if event.health:
        lines.extend(["", event.health])
    if event.gpu_name:
        lines.extend(["", "GPU:", event.gpu_name])
    if event.vram_used_mb is not None and event.vram_total_mb:
        used = event.vram_used_mb / 1024
        total = event.vram_total_mb / 1024
        lines.extend(["", "VRAM:", f"{used:.1f} / {total:.1f} GB"])
    if event.eta_seconds is not None and event.status == "ACTIVE":
        lines.extend(["", "ETA (this node):", _clock(event.eta_seconds)])
    if event.status == "FAILED":
        lines.extend(["", "FAILED"])
        if event.error_node:
            lines.extend(["Node:", event.error_node])
        if event.error_type:
            lines.extend(["Node type:", event.error_type])
        if event.error_message:
            lines.extend(["Message:", event.error_message])
    if event.preview_kind == "comfy":
        lines.extend(["", "ComfyUI AI preview — not final output"])
    elif event.preview_kind == "completed":
        lines.extend(["", "Latest completed frame"])
    return "\n".join(lines)


def stamp_log(message: str, *, now: float | None = None) -> str:
    when = time.localtime(now)
    return f"[{time.strftime('%H:%M:%S', when)}] {message}"


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _bar(percent: float) -> str:
    bounded = max(0.0, min(100.0, percent))
    filled = int(round(16 * bounded / 100))
    return f"{'█' * filled}{'░' * (16 - filled)}  {bounded:.0f}%"


def _clock(seconds: float) -> str:
    whole = max(0, int(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def shorten_gpu_name(name: str) -> str:
    text = name.replace("cuda:0", "").replace("cudaMallocAsync", "")
    text = text.replace("NVIDIA GeForce", "").replace(":", " ")
    return " ".join(text.split())
