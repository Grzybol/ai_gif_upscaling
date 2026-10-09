"""Progress events stay tied to one prompt and do not invent finished frames."""

import json
import struct

from midnight_upscale.progress import (
    PREVIEW_IMAGE,
    PREVIEW_IMAGE_WITH_METADATA,
    ProgressBus,
    bind_bus,
    format_status,
    parse_preview_bytes,
    report,
    reset_bus,
)


def _message(event: str, data: dict[str, object]) -> str:
    return json.dumps({"type": event, "data": data})


def test_events_for_another_prompt_are_ignored() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.set_nodes({"10": "KSampler"})
    bus.apply_socket(_message("executing", {"node": "10", "prompt_id": "prompt-b"}))
    bus.apply_socket(
        _message("progress", {"value": 4, "max": 20, "prompt_id": "prompt-b", "node": "10"})
    )
    event = bus.snapshot()
    assert event.comfy_node_type == ""
    assert event.node_step is None
    assert event.frames_done is None


def test_executing_then_progress_tracks_the_node_not_frames() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.set_nodes({"9": "SeedVR2Conditioning", "10": "KSampler"})
    bus.apply_socket(_message("execution_start", {"prompt_id": "prompt-a"}))
    bus.apply_socket(_message("executing", {"node": "10", "prompt_id": "prompt-a"}))
    bus.apply_socket(
        _message("progress", {"value": 7, "max": 20, "prompt_id": "prompt-a", "node": "10"})
    )
    event = bus.snapshot()
    assert event.stage == "SeedVR2 processing"
    assert event.comfy_node_type == "KSampler"
    assert event.node_step == 7
    assert event.node_steps_total == 20
    assert event.frames_done is None
    assert event.status == "ACTIVE"


def test_execution_success_and_error_are_distinct() -> None:
    ok = ProgressBus()
    ok.assign_prompt("prompt-a")
    ok.apply_socket(_message("execution_success", {"prompt_id": "prompt-a"}))
    assert ok.snapshot().status != "FAILED"
    assert ok.comfy_failed == ""

    failed = ProgressBus()
    failed.assign_prompt("prompt-a")
    failed.apply_socket(
        _message(
            "execution_error",
            {
                "prompt_id": "prompt-a",
                "node_id": "12",
                "node_type": "VAEDecodeTiled",
                "exception_message": "out of memory",
                "exception_type": "RuntimeError",
            },
        )
    )
    event = failed.snapshot()
    assert event.status == "FAILED"
    assert event.error_node == "12"
    assert event.error_type == "VAEDecodeTiled"
    assert event.error_message == "out of memory"
    assert failed.comfy_failed == "out of memory"


def test_interrupt_marks_cancelled() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.apply_socket(
        _message(
            "execution_interrupted",
            {"prompt_id": "prompt-a", "node_id": "10", "node_type": "KSampler"},
        )
    )
    assert bus.snapshot().status == "CANCELLED"


def test_socket_reconnect_is_counted_and_does_not_clear_the_prompt() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.note_socket(False)
    bus.note_socket(True)
    event = bus.snapshot()
    assert bus.reconnects == 1
    assert event.ws_alive is True
    assert event.comfy_prompt_id == "prompt-a"


def test_preview_metadata_from_another_prompt_is_dropped() -> None:
    own = json.dumps({"prompt_id": "prompt-a"}).encode()
    other = json.dumps({"prompt_id": "prompt-b"}).encode()
    own_blob = struct.pack(">II", PREVIEW_IMAGE_WITH_METADATA, len(own)) + own + b"jpeg-a"
    other_blob = struct.pack(">II", PREVIEW_IMAGE_WITH_METADATA, len(other)) + other + b"jpeg-b"
    assert parse_preview_bytes(own_blob, "prompt-a") == b"jpeg-a"
    assert parse_preview_bytes(other_blob, "prompt-a") is None
    unlabeled = struct.pack(">II", PREVIEW_IMAGE, 1) + b"jpeg-raw"
    assert parse_preview_bytes(unlabeled, "") is None
    assert parse_preview_bytes(unlabeled, "prompt-a") == b"jpeg-raw"


def test_sampler_steps_do_not_become_pipeline_percent_or_finished_frames() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.frames_total = 120
    bus.set_nodes({"1": "LoadVideo", "10": "KSampler", "12": "VAEDecodeTiled"})
    bus.apply_socket(_message("execution_start", {"prompt_id": "prompt-a"}))
    bus.apply_socket(_message("executing", {"node": "10", "prompt_id": "prompt-a"}))
    bus.apply_socket(
        _message("progress", {"value": 9, "max": 20, "prompt_id": "prompt-a", "node": "10"})
    )
    event = bus.snapshot()
    text = format_status(event)
    assert event.overall_percent is None
    assert "9 / 20" in text
    assert "Frames completed" not in text
    assert "Frames submitted" in text
    assert "█" in text
    assert event.frames_done is None


def test_a_node_without_progress_is_not_given_a_percent() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.set_nodes({"4": "SeedVR2Preprocess"})
    bus.apply_socket(_message("executing", {"node": "4", "prompt_id": "prompt-a"}))
    text = format_status(bus.snapshot())
    assert "Current node:" in text
    assert "SeedVR2Preprocess" in text
    assert "Not reported by ComfyUI" in text
    assert "█" not in text


def test_alpha_resize_is_not_a_completed_output_preview() -> None:
    bus = ProgressBus()
    bus.update_stage(
        "Upscale alpha",
        message="Alpha resize 1/5",
        frames_done=1,
        frames_total=5,
        frames_kind="completed",
        preview_path="alpha.png",
    )
    event = bus.snapshot()
    assert event.preview_kind == ""
    assert event.preview_path == ""
    assert "Frames completed:" in format_status(event)


def test_report_records_real_frame_progress() -> None:
    bus = ProgressBus()
    token = bind_bus(bus)
    try:
        report(
            "Decode GIF",
            message="Decoded frame 2/5",
            frames_done=2,
            frames_total=5,
            frames_kind="completed",
        )
    finally:
        reset_bus(token)
    event = bus.snapshot()
    assert event.frames_done == 2
    assert event.frames_total == 5
    assert event.frames_kind == "completed"


def test_completed_preview_is_separate_from_a_comfy_preview() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.update_stage(
        "Decode upscaled RGB",
        message="RGB frame 1/5",
        frames_done=1,
        frames_total=5,
        frames_kind="completed",
        preview_path="frame.png",
    )
    event = bus.snapshot()
    assert event.preview_kind == "completed"
    assert event.preview_path == "frame.png"
    blob = struct.pack(">II", PREVIEW_IMAGE, 1) + b"jpeg"
    bus.apply_socket(blob)
    comfy = bus.snapshot()
    assert comfy.preview_kind == "comfy"
    assert comfy.preview_bytes == b"jpeg"


def test_no_recent_progress_still_says_the_job_is_running() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.last_comfy_at = 0.0
    bus.note_queue("RUNNING", ws_alive=True)
    event = bus.snapshot()
    assert event.status == "ACTIVE — no recent progress event"
    assert "running queue" in event.health
    assert "HUNG" not in event.health


def test_missing_queue_entry_without_history_is_unknown() -> None:
    bus = ProgressBus()
    bus.assign_prompt("prompt-a")
    bus.note_queue("UNKNOWN", ws_alive=True, history_present=False)
    event = bus.snapshot()
    assert event.status == "UNKNOWN"
    assert "history" in event.health
