"""ComfyUI HTTP client and configurable workflow mapping.

Node class types come from the workflow file the user points at. They are
checked against ``GET /object_info`` before a prompt is submitted. This module
does not invent SeedVR2 node names and it does not submit the example template
while its class types are still placeholders.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from midnight_upscale.progress import JobCancelled, current_bus
from midnight_upscale.utils import (
    ComfyError,
    WorkflowConfigError,
    as_node_id,
    tail_text,
)

logger = logging.getLogger(__name__)

PLACEHOLDER_PREFIX = "REPLACE_"

# Names confirmed in numz/ComfyUI-SeedVR2_VideoUpscaler (node_id in define_schema).
# Used only to report whether this install has those nodes. Never submitted unless
# the user's own workflow JSON contains them.
KNOWN_SEEDVR2_CLASS_TYPES = (
    "SeedVR2VideoUpscaler",
    "SeedVR2LoadDiTModel",
    "SeedVR2LoadVAEModel",
)


def load_workflow(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise WorkflowConfigError(f"Workflow file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowConfigError(f"Workflow {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkflowConfigError(f"Workflow {path} must be a JSON object of nodes")
    return sanitize_workflow(data)


def sanitize_workflow(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if str(key).startswith("_"):
            continue
        if not isinstance(value, dict):
            raise WorkflowConfigError(f"Workflow node {key} is not an object")
        node = {
            str(item): item_value
            for item, item_value in value.items()
            if not str(item).startswith("_")
        }
        if "class_type" not in node or "inputs" not in node:
            raise WorkflowConfigError(f"Workflow node {key} must contain class_type and inputs")
        if not isinstance(node["inputs"], dict):
            raise WorkflowConfigError(f"Workflow node {key} inputs must be an object")
        nodes[str(key)] = node
    if not nodes:
        raise WorkflowConfigError("Workflow has no nodes")
    return nodes


def placeholder_nodes(workflow: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node_id, node in workflow.items():
        class_type = str(node.get("class_type", ""))
        if class_type.startswith(PLACEHOLDER_PREFIX):
            found.append((node_id, class_type))
    return found


def assert_workflow_configured(workflow: dict[str, dict[str, Any]], workflow_path: Path) -> None:
    placeholders = placeholder_nodes(workflow)
    if not placeholders:
        return
    lines = "\n".join(f"  node {node_id}: {class_type}" for node_id, class_type in placeholders)
    raise WorkflowConfigError(
        f"{workflow_path} is still the example template. These class types are placeholders "
        "and were not submitted:\n"
        f"{lines}\n"
        "Export your ComfyUI graph in API format, point seedvr2.workflow at that file, "
        "and set the node ids in config.yaml. Run `midnight-upscale comfy-info` to list "
        "the nodes this ComfyUI instance actually has. See workflows/README.md."
    )


def assert_nodes_available(
    workflow: dict[str, dict[str, Any]],
    object_info: dict[str, Any],
) -> None:
    missing = [
        (node_id, str(node.get("class_type")))
        for node_id, node in workflow.items()
        if str(node.get("class_type")) not in object_info
    ]
    if not missing:
        return
    lines = "\n".join(f"  node {node_id}: {class_type}" for node_id, class_type in missing)
    raise WorkflowConfigError(
        "The workflow references ComfyUI nodes that are not available on this instance:\n"
        f"{lines}\n"
        "Refusing to continue. Run `midnight-upscale comfy-info` and map class_type values "
        "to names in that file. See workflows/README.md."
    )


def schema_has_input(object_info: dict[str, Any], class_type: str, field: str) -> bool:
    """True when ComfyUI ``/object_info`` lists ``field`` on ``class_type``."""

    node = object_info.get(class_type)
    if not isinstance(node, dict):
        return False
    groups = node.get("input")
    if not isinstance(groups, dict):
        return False
    for section in ("required", "optional"):
        inputs = groups.get(section)
        if isinstance(inputs, dict) and field in inputs:
            return True
    return False


def resolve_temporal_overlap_target(
    workflow: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    object_info: dict[str, Any],
) -> tuple[str, str] | None:
    """Return the node and field to write, or None when the node has no such input.

    A missing input is not created. The caller warns and leaves the node default.
    """

    configured = settings.get("temporal_overlap") or {}
    node_id = as_node_id(configured.get("node_id"))
    if not node_id:
        node_id = as_node_id((settings.get("scale") or {}).get("node_id"))
    field = str(configured.get("field") or "temporal_overlap")
    if not node_id or node_id not in workflow or not field:
        return None
    node = workflow[node_id]
    class_type = str(node.get("class_type") or "")
    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
    if schema_has_input(object_info, class_type, field) or field in inputs:
        return node_id, field
    return None


def set_node_input(
    workflow: dict[str, dict[str, Any]], node_id: str, field: str, value: Any
) -> None:
    if not node_id:
        return
    if node_id not in workflow:
        raise WorkflowConfigError(
            f"Config maps node id {node_id!r}, but that id is not in the workflow. "
            f"Node ids present: {', '.join(workflow)}"
        )
    inputs = workflow[node_id].get("inputs")
    if not isinstance(inputs, dict):
        raise WorkflowConfigError(f"Node {node_id} inputs must be an object")
    inputs[field] = value


def resolution_for_scale(mode: str, scale: int, width: int, height: int) -> tuple[int, bool]:
    """Return the value to write into the scale node, and whether it was rounded.

    ``shortest_edge`` matches SeedVR2VideoUpscaler, whose ``resolution`` input is
    the target length of the shorter side in pixels and wants an even number.
    The returned flag is true when that even-number adjustment changed the value.
    """

    if mode == "shortest_edge":
        value = min(width, height) * scale
        if value % 2:
            return value + 1, True
        return value, False
    if mode == "scale_factor":
        return scale, False
    if mode == "target_width":
        return width * scale, False
    if mode == "target_height":
        return height * scale, False
    raise WorkflowConfigError(
        f"Unknown scale mode {mode!r}. "
        "Use shortest_edge, scale_factor, target_width, or target_height."
    )


def apply_seedvr2_overrides(
    workflow: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    *,
    input_path: str | None,
    output_path: str | None,
    uploaded_name: str | None,
    scale: int,
    width: int,
    height: int,
    batch_size: int,
    require_controls: bool = True,
    temporal_overlap: int | None = None,
    temporal_overlap_node: str = "",
    temporal_overlap_field: str = "",
) -> int:
    """Write mapped inputs. Returns the numeric resolution/scale value that was set.

    ``require_controls`` is true for SeedVR2, where scale and batch size must be
    written into the graph. The frame-upscale fallback can omit those widgets.
    """

    workflow_copy_inputs_ready(workflow)
    input_cfg = settings.get("input") or {}
    input_id = as_node_id(input_cfg.get("node_id"))
    input_field = str(input_cfg.get("field") or "")
    input_mode = str(input_cfg.get("mode") or "directory")
    if not input_id or not input_field:
        raise WorkflowConfigError("input.node_id and input.field are required")
    if input_mode == "directory":
        if not input_path:
            raise WorkflowConfigError("Directory input mode requires an input path")
        set_node_input(workflow, input_id, input_field, input_path)
    elif input_mode == "upload_image":
        if not uploaded_name:
            raise WorkflowConfigError("upload_image mode requires an uploaded filename")
        set_node_input(workflow, input_id, input_field, uploaded_name)
    else:
        raise WorkflowConfigError(
            f"Unknown input mode {input_mode!r}. Use directory or upload_image."
        )

    model_cfg = settings.get("model") or {}
    model_value = model_cfg.get("value")
    if as_node_id(model_cfg.get("node_id")) and model_value not in (None, ""):
        set_node_input(
            workflow,
            as_node_id(model_cfg.get("node_id")),
            str(model_cfg.get("field") or "model"),
            model_value,
        )

    scale_cfg = settings.get("scale") or {}
    scale_id = as_node_id(scale_cfg.get("node_id"))
    scale_field = str(scale_cfg.get("field") or "")
    resolution = 0
    if scale_id and scale_field:
        resolution, _rounded = resolution_for_scale(
            str(scale_cfg.get("mode") or "shortest_edge"),
            scale,
            width,
            height,
        )
        set_node_input(workflow, scale_id, scale_field, resolution)
    elif require_controls:
        raise WorkflowConfigError(
            "scale.node_id is empty, so --scale cannot be applied to the workflow."
        )

    batch_cfg = settings.get("batch_size") or {}
    batch_id = as_node_id(batch_cfg.get("node_id"))
    batch_field = str(batch_cfg.get("field") or "")
    if batch_id and batch_field:
        set_node_input(workflow, batch_id, batch_field, batch_size)
    elif require_controls:
        raise WorkflowConfigError("batch_size.node_id is empty, so --batch-size cannot be applied.")

    if temporal_overlap is not None and temporal_overlap_node and temporal_overlap_field:
        set_node_input(
            workflow,
            temporal_overlap_node,
            temporal_overlap_field,
            temporal_overlap,
        )

    output_cfg = settings.get("output") or {}
    output_mode = str(output_cfg.get("mode") or "directory")
    output_id = as_node_id(output_cfg.get("node_id"))
    output_field = str(output_cfg.get("field") or "")
    if output_mode == "directory":
        if not output_id or not output_field or not output_path:
            raise WorkflowConfigError(
                "directory output mode requires output.node_id, field, and a path"
            )
        set_node_input(workflow, output_id, output_field, output_path)
    elif output_mode == "history":
        if not output_id:
            raise WorkflowConfigError("history output mode requires output.node_id")
        # History mode reads images back from /history. Do not stuff a directory
        # into filename_prefix unless the config explicitly asks for that.
        if output_cfg.get("write_value") and output_field and output_path:
            set_node_input(workflow, output_id, output_field, output_path)
    else:
        raise WorkflowConfigError(f"Unknown output mode {output_mode!r}. Use directory or history.")
    return resolution


def workflow_copy_inputs_ready(workflow: dict[str, dict[str, Any]]) -> None:
    for node_id, node in workflow.items():
        if "inputs" not in node or not isinstance(node["inputs"], dict):
            raise WorkflowConfigError(f"Node {node_id} is missing inputs")


def known_seedvr2_status(object_info: dict[str, Any]) -> list[str]:
    lines = []
    for name in KNOWN_SEEDVR2_CLASS_TYPES:
        state = "present" if name in object_info else "not installed"
        lines.append(f"{name}: {state}")
    return lines


def unwrap_history(payload: dict[str, Any], prompt_id: str) -> dict[str, Any] | None:
    if not payload:
        return None
    entry = payload.get(prompt_id)
    if isinstance(entry, dict):
        return entry
    if "outputs" in payload or "status" in payload:
        return payload
    return None


def history_error(entry: dict[str, Any]) -> str | None:
    status = entry.get("status") or {}
    if not isinstance(status, dict):
        return f"Unexpected ComfyUI status: {status!r}"
    messages = status.get("messages") or []
    chunks: list[str] = []
    for item in messages:
        if isinstance(item, (list, tuple)) and item and item[0] == "execution_error":
            detail = item[1] if len(item) > 1 else {}
            if isinstance(detail, dict):
                chunks.append(
                    f"node {detail.get('node_id')}: {detail.get('exception_type')}: "
                    f"{detail.get('exception_message')}"
                )
            else:
                chunks.append(str(detail))
    status_str = status.get("status_str")
    if status_str == "error" or chunks:
        return "; ".join(chunks) or json.dumps(status)
    if status.get("completed") and status_str not in (None, "success"):
        return json.dumps(status)
    return None


def history_is_complete(entry: dict[str, Any]) -> bool:
    status = entry.get("status") or {}
    if not isinstance(status, dict):
        return False
    return bool(status.get("completed")) or status.get("status_str") == "success"


def images_from_history(entry: dict[str, Any], node_id: str) -> list[dict[str, Any]]:
    outputs = entry.get("outputs") or {}
    if not isinstance(outputs, dict):
        raise ComfyError("ComfyUI history outputs were not an object")
    node_out = outputs.get(node_id)
    if not isinstance(node_out, dict):
        present = ", ".join(outputs) or "(none)"
        raise ComfyError(
            f"ComfyUI finished but node {node_id} produced no outputs. Output nodes: {present}"
        )
    images = node_out.get("images") or node_out.get("gifs") or []
    if not isinstance(images, list) or not images:
        raise ComfyError(
            f"Node {node_id} did not return images. Keys: {', '.join(node_out) or '(none)'}"
        )
    return images


def format_prompt_rejection(body: dict[str, Any]) -> str:
    parts: list[str] = []
    error = body.get("error")
    if isinstance(error, dict):
        parts.append(str(error.get("message") or error.get("type") or error))
    elif error:
        parts.append(str(error))
    node_errors = body.get("node_errors") or {}
    if isinstance(node_errors, dict):
        for node_id, info in node_errors.items():
            if isinstance(info, dict):
                for item in info.get("errors") or []:
                    if isinstance(item, dict):
                        message = f"{item.get('message', '')} {item.get('details', '')}".strip()
                        parts.append(f"node {node_id}: {message}")
                    else:
                        parts.append(f"node {node_id}: {item}")
            else:
                parts.append(f"node {node_id}: {info}")
    return "\n".join(parts) or json.dumps(body)[:2000]


def _queue_contains(items: object, prompt_id: str) -> bool:
    if not isinstance(items, list):
        return False
    for item in items:
        if isinstance(item, list) and len(item) > 1 and str(item[1]) == prompt_id:
            return True
    return False


class ComfyClient:
    def __init__(self, base_url: str, *, timeout_sec: float, poll_interval_sec: float) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ComfyError(
                f"ComfyUI URL {base_url!r} is not an http(s) URL. Example: http://127.0.0.1:8188"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.client_id = str(uuid.uuid4())
        self.ws_alive = False
        self._progress_bus: Any = None
        self._ws_stop = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._client = httpx.Client(
            base_url=self.base_url, timeout=httpx.Timeout(60.0, connect=10.0)
        )

    def close(self) -> None:
        self._ws_stop.set()
        thread = self._ws_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._client.close()

    def ensure_socket(self) -> None:
        """Listen on the same client id used to submit the prompt. Reconnects do not resubmit."""

        # The socket thread does not inherit this thread's context, so keep the bus on the client.
        self._progress_bus = current_bus()
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        self._ws_stop.clear()
        self._ws_thread = threading.Thread(
            target=self._socket_loop,
            name="midnight-comfy-ws",
            daemon=True,
        )
        self._ws_thread.start()
        for _ in range(20):
            if self.ws_alive:
                return
            time.sleep(0.1)

    def _socket_loop(self) -> None:
        try:
            import websocket
        except ImportError:
            logger.warning(
                "websocket-client is not installed. Progress falls back to queue checks."
            )
            return
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        http_base = self.base_url.split("://", 1)[1]
        url = f"{scheme}://{http_base}/ws?clientId={self.client_id}"
        connected_once = False
        while not self._ws_stop.is_set():
            socket = websocket.WebSocket()
            socket.settimeout(5)
            try:
                socket.connect(url, timeout=10)
                self.ws_alive = True
                bus = self._progress_bus
                if bus is not None:
                    bus.note_socket(True)
                if connected_once:
                    logger.info("ComfyUI WebSocket reconnected for client %s.", self.client_id[:8])
                connected_once = True
                while not self._ws_stop.is_set():
                    try:
                        message = socket.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if message is None or message == "":
                        break
                    bus = self._progress_bus
                    if bus is not None:
                        bus.apply_socket(message)
            except Exception as exc:
                self.ws_alive = False
                bus = self._progress_bus
                if bus is not None:
                    bus.note_socket(False)
                if self._ws_stop.is_set():
                    break
                logger.info("ComfyUI WebSocket disconnected (%s). Reconnecting.", exc)
                if self._ws_stop.wait(2):
                    break
            finally:
                self.ws_alive = False
                try:
                    socket.close()
                except Exception:
                    pass

    def interrupt(self, prompt_id: str) -> None:
        self._request("POST", "/interrupt", json={"prompt_id": prompt_id})

    def delete_pending(self, prompt_id: str) -> None:
        self._request("POST", "/queue", json={"delete": [prompt_id]})

    def prompt_queue_state(self, prompt_id: str) -> str:
        response = self._request("GET", "/queue")
        payload = self._json(response)
        if not isinstance(payload, dict):
            return "UNKNOWN"
        if _queue_contains(payload.get("queue_running"), prompt_id):
            return "RUNNING"
        if _queue_contains(payload.get("queue_pending"), prompt_id):
            return "WAITING"
        return "UNKNOWN"

    def gpu_sample(self) -> tuple[str, float | None, float | None]:
        from midnight_upscale.progress import shorten_gpu_name

        payload = self.system_stats()
        devices = payload.get("devices")
        if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
            return "", None, None
        device = devices[0]
        total = device.get("vram_total")
        free = device.get("vram_free")
        if not isinstance(total, (int, float)) or not isinstance(free, (int, float)):
            return shorten_gpu_name(str(device.get("name") or "")), None, None
        used_mb = max(0.0, (float(total) - float(free)) / (1024 * 1024))
        total_mb = float(total) / (1024 * 1024)
        return shorten_gpu_name(str(device.get("name") or "")), used_mb, total_mb

    def object_info(self) -> dict[str, Any]:
        response = self._request("GET", "/object_info")
        payload = self._json(response)
        if not isinstance(payload, dict):
            raise ComfyError("ComfyUI /object_info did not return an object")
        return payload

    def system_stats(self) -> dict[str, Any]:
        response = self._request("GET", "/system_stats")
        payload = self._json(response)
        if not isinstance(payload, dict):
            raise ComfyError("ComfyUI /system_stats did not return an object")
        return payload

    def upload_image(self, path: Path, *, subfolder: str = "") -> str:
        return self.upload_input(path, content_type="image/png", subfolder=subfolder)

    def upload_input(self, path: Path, *, content_type: str, subfolder: str = "") -> str:
        try:
            with path.open("rb") as handle:
                response = self._client.post(
                    "/upload/image",
                    data={"overwrite": "true", "type": "input", "subfolder": subfolder},
                    files={"image": (path.name, handle, content_type)},
                    timeout=httpx.Timeout(180.0, connect=10.0),
                )
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc
        self._raise_for_status(response)
        payload = self._json(response)
        name = payload.get("name") if isinstance(payload, dict) else None
        if not name:
            raise ComfyError(f"ComfyUI upload of {path.name} did not return a filename")
        return str(name)

    def run_workflow(self, workflow: dict[str, dict[str, Any]]) -> dict[str, Any]:
        bus = current_bus()
        if bus is not None:
            self.ensure_socket()
        prompt_id = self.submit(workflow)
        if bus is not None:
            bus.assign_prompt(prompt_id)
        return self.wait(prompt_id)

    def submit(self, workflow: dict[str, dict[str, Any]]) -> str:
        response = self._request(
            "POST",
            "/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
        )
        payload = self._json(response)
        if not isinstance(payload, dict):
            raise ComfyError("ComfyUI /prompt did not return an object")
        node_errors = payload.get("node_errors") or {}
        if node_errors or "prompt_id" not in payload:
            raise ComfyError(
                "ComfyUI rejected the workflow and it was not queued.\n"
                + format_prompt_rejection(payload)
            )
        return str(payload["prompt_id"])

    def wait(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_sec
        last_health = 0.0
        while True:
            bus = current_bus()
            if bus is not None and bus.cancel_requested():
                self._cancel_prompt(prompt_id)
                bus.mark_cancelled(f"Cancelled ComfyUI prompt {prompt_id[:8]}.")
                raise JobCancelled(
                    f"Cancelled ComfyUI prompt {prompt_id}. Partial output was not finalized."
                )
            if bus is not None and bus.comfy_failed:
                raise ComfyError(
                    f"ComfyUI prompt {prompt_id} failed in node {bus.error_node or '?'} "
                    f"({bus.error_type or bus.node_type or 'node'}): {bus.comfy_failed}"
                )
            now = time.monotonic()
            if bus is not None and now - last_health >= 8:
                last_health = now
                self._note_health(prompt_id, bus)
            try:
                response = self._client.request("GET", f"/history/{prompt_id}")
            except httpx.ReadTimeout:
                if time.monotonic() > deadline:
                    raise ComfyError(
                        f"Timed out after {self.timeout_sec:.0f}s "
                        f"waiting for ComfyUI prompt {prompt_id}. "
                        "The prompt was not treated as successful and no frames were invented."
                    )
                logger.info(
                    "ComfyUI is still inside a running node, so the history poll timed out. "
                    "Waiting continues for prompt %s.",
                    prompt_id,
                )
                self._pause_for_cancel(bus)
                continue
            except httpx.HTTPError as exc:
                raise self._unreachable(exc) from exc
            self._raise_for_status(response)
            payload = self._json(response)
            if not isinstance(payload, dict):
                raise ComfyError("ComfyUI history response was not an object")
            entry = unwrap_history(payload, prompt_id)
            if entry is not None:
                error = history_error(entry)
                if error:
                    raise ComfyError(f"ComfyUI prompt {prompt_id} failed: {error}")
                if history_is_complete(entry):
                    if bus is not None and bus.cancel_requested():
                        continue
                    return entry
            if time.monotonic() > deadline:
                waited = (
                    f"Timed out after {self.timeout_sec:.0f}s "
                    f"waiting for ComfyUI prompt {prompt_id}."
                )
                raise ComfyError(
                    waited
                    + " The prompt was not treated as successful and no frames were invented."
                )
            self._pause_for_cancel(bus)

    def _pause_for_cancel(self, bus: Any) -> None:
        if bus is not None:
            bus.cancel_event.wait(self.poll_interval_sec)
            return
        time.sleep(self.poll_interval_sec)

    def _note_health(self, prompt_id: str, bus: Any) -> None:
        try:
            state = self.prompt_queue_state(prompt_id)
        except ComfyError:
            state = "UNKNOWN"
        history_present = False
        if state == "UNKNOWN":
            history_present = self._history_has_prompt(prompt_id)
        bus.note_queue(state, ws_alive=self.ws_alive, history_present=history_present)
        try:
            name, used, total = self.gpu_sample()
        except ComfyError:
            return
        if name:
            bus.note_gpu(name, used, total)

    def _history_has_prompt(self, prompt_id: str) -> bool:
        try:
            response = self._client.request("GET", f"/history/{prompt_id}")
        except httpx.HTTPError:
            return False
        if response.status_code >= 400:
            return False
        payload = self._json(response)
        return isinstance(payload, dict) and unwrap_history(payload, prompt_id) is not None

    def cancel_tracked(self, prompt_id: str) -> bool:
        """Cancel one prompt. Prefer POST /api/jobs/{id}/cancel. Never interrupt without an id."""

        if not prompt_id:
            logger.info("Cancel requested before a ComfyUI prompt was queued.")
            logger.info("Cancel acknowledged: false")
            return False
        logger.info("Cancel requested for %s", prompt_id)
        acknowledged = False
        try:
            response = self._request("POST", f"/api/jobs/{prompt_id}/cancel", json={})
            payload = self._json(response)
            acknowledged = isinstance(payload, dict) and bool(payload.get("cancelled"))
        except ComfyError as exc:
            logger.info(
                "Targeted job cancel was not accepted (%s). Trying the queue fallback.",
                exc,
            )
            acknowledged = self._cancel_fallback(prompt_id)
        logger.info("Cancel acknowledged: %s", str(acknowledged).lower())
        return acknowledged

    def _cancel_fallback(self, prompt_id: str) -> bool:
        try:
            state = self.prompt_queue_state(prompt_id)
        except ComfyError:
            return False
        try:
            if state == "WAITING":
                self.delete_pending(prompt_id)
                return True
            if state == "RUNNING":
                self.interrupt(prompt_id)
                return True
        except ComfyError as exc:
            logger.warning("Could not cancel prompt %s: %s", prompt_id, exc)
        return False

    def _cancel_prompt(self, prompt_id: str) -> None:
        self.cancel_tracked(prompt_id)

    def download(self, image_info: dict[str, Any], dest: Path) -> None:
        filename = image_info.get("filename")
        if not filename:
            raise ComfyError(f"ComfyUI image info has no filename: {image_info}")
        try:
            response = self._client.get(
                "/view",
                params={
                    "filename": filename,
                    "subfolder": image_info.get("subfolder") or "",
                    "type": image_info.get("type") or "output",
                },
            )
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc
        self._raise_for_status(response)
        if not response.content:
            raise ComfyError(f"ComfyUI returned an empty file for {filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc
        self._raise_for_status(response)
        return response

    def _unreachable(self, exc: httpx.HTTPError) -> ComfyError:
        return ComfyError(
            f"Could not reach ComfyUI at {self.base_url}: {exc}. "
            "Start ComfyUI and pass --comfy-url if it is not on http://127.0.0.1:8188."
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        raise ComfyError(
            f"ComfyUI returned HTTP {response.status_code} for {response.request.url}.\n"
            f"{tail_text(response.text)}"
        )

    @staticmethod
    def _json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ComfyError(
                f"ComfyUI returned non-JSON from {response.request.url}: {tail_text(response.text)}"
            ) from exc
