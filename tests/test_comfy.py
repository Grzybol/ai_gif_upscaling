import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from midnight_upscale.cli import main
from midnight_upscale.comfy import (
    ComfyClient,
    apply_seedvr2_overrides,
    assert_nodes_available,
    assert_workflow_configured,
    format_prompt_rejection,
    history_error,
    load_workflow,
    resolution_for_scale,
    resolve_temporal_overlap_target,
)
from midnight_upscale.pipeline import prepare_asset
from midnight_upscale.seedvr2 import (
    run_frame_upscale,
    run_seedvr2,
    validate_seedvr2_batch_size,
    validate_temporal_overlap,
)
from midnight_upscale.utils import (
    ComfyError,
    PipelineError,
    ValidationError,
    WorkflowConfigError,
    list_indexed_frames,
)
from tests.conftest import character_frame, save_rgba_gif

ROOT = Path(__file__).parents[1]


class DirectoryUpscaler:
    def __init__(self, nodes: dict[str, Any], scale: int) -> None:
        self.nodes = nodes
        self.scale = scale
        self.calls: list[dict[str, Any]] = []

    def object_info(self) -> dict[str, Any]:
        return self.nodes

    def upload_image(self, path: Path, *, subfolder: str = "") -> str:
        raise AssertionError("directory mode must not upload frames")

    def download(self, image_info: dict[str, Any], dest: Path) -> None:
        raise AssertionError("directory mode must not download history images")

    def run_workflow(self, workflow: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.calls.append(json.loads(json.dumps(workflow)))
        source_dir = Path(workflow["1"]["inputs"]["directory"])
        output_dir = Path(workflow["5"]["inputs"]["output_path"])
        assert "alpha" not in source_dir.parts
        assert source_dir.name == "input"
        for path in list_indexed_frames(source_dir):
            with Image.open(path) as image:
                resized = image.resize(
                    (image.width * self.scale, image.height * self.scale),
                    Image.Resampling.NEAREST,
                )
                resized.save(output_dir / path.name)
        return {"status": {"completed": True, "status_str": "success"}, "outputs": {}}


def _workflow(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "1": {"class_type": "TestLoader", "inputs": {"directory": ""}},
                "4": {
                    "class_type": "TestUpscaler",
                    "inputs": {"image": ["1", 0], "resolution": 0, "batch_size": 1},
                },
                "5": {
                    "class_type": "TestSaver",
                    "inputs": {"images": ["4", 0], "output_path": ""},
                },
            }
        ),
        encoding="utf-8",
    )


def _config(path: Path, workflow: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "seedvr2:",
                f"  workflow: {workflow.as_posix()}",
                "  chunking: windows",
                "  input:",
                '    node_id: "1"',
                "    field: directory",
                "    mode: directory",
                "  scale:",
                '    node_id: "4"',
                "    field: resolution",
                "    mode: shortest_edge",
                "  batch_size:",
                '    node_id: "4"',
                "    field: batch_size",
                "  output:",
                '    node_id: "5"',
                "    field: output_path",
                "    mode: directory",
            ]
        ),
        encoding="utf-8",
    )


def test_example_workflow_is_rejected_without_contacting_comfyui() -> None:
    workflow = load_workflow(ROOT / "workflows" / "seedvr2.example.json")
    with pytest.raises(WorkflowConfigError, match="placeholder"):
        assert_workflow_configured(workflow, ROOT / "workflows" / "seedvr2.example.json")


def test_missing_node_is_reported_and_not_ignored() -> None:
    workflow = {
        "4": {"class_type": "SeedVR2VideoUpscaler", "inputs": {}},
        "9": {"class_type": "NotInstalled", "inputs": {}},
    }
    with pytest.raises(WorkflowConfigError, match="NotInstalled"):
        assert_nodes_available(workflow, {"SeedVR2VideoUpscaler": {}})


def test_overrides_set_scale_and_batch_without_dropping_links() -> None:
    workflow = {
        "1": {"class_type": "TestLoader", "inputs": {"directory": ""}},
        "4": {
            "class_type": "TestUpscaler",
            "inputs": {"image": ["1", 0], "resolution": 1, "batch_size": 1},
        },
        "5": {"class_type": "TestSaver", "inputs": {"images": ["4", 0], "output_path": ""}},
    }
    settings = {
        "input": {"node_id": "1", "field": "directory", "mode": "directory"},
        "scale": {"node_id": "4", "field": "resolution", "mode": "shortest_edge"},
        "batch_size": {"node_id": "4", "field": "batch_size"},
        "output": {"node_id": "5", "field": "output_path", "mode": "directory"},
    }
    resolution = apply_seedvr2_overrides(
        workflow,
        settings,
        input_path="D:/work/rgb",
        output_path="D:/work/out",
        uploaded_name=None,
        scale=2,
        width=6,
        height=10,
        batch_size=5,
    )
    assert resolution == 12
    assert workflow["4"]["inputs"]["image"] == ["1", 0]
    assert workflow["4"]["inputs"]["resolution"] == 12
    assert workflow["4"]["inputs"]["batch_size"] == 5
    assert workflow["1"]["inputs"]["directory"] == "D:/work/rgb"


def test_odd_shortest_edge_is_rounded_up() -> None:
    value, rounded = resolution_for_scale("shortest_edge", 1, 3, 5)
    assert (value, rounded) == (4, True)
    value, rounded = resolution_for_scale("scale_factor", 2, 3, 5)
    assert (value, rounded) == (2, False)


def test_prompt_and_history_errors_are_visible() -> None:
    text = format_prompt_rejection(
        {
            "error": {"message": "bad prompt"},
            "node_errors": {"4": {"errors": [{"message": "missing model", "details": "dit"}]}},
        }
    )
    assert "bad prompt" in text
    assert "missing model" in text
    error = history_error(
        {
            "status": {
                "status_str": "error",
                "completed": True,
                "messages": [
                    [
                        "execution_error",
                        {
                            "node_id": "4",
                            "exception_type": "RuntimeError",
                            "exception_message": "model file missing",
                        },
                    ]
                ],
            }
        }
    )
    assert error is not None
    assert "model file missing" in error


def test_seedvr2_batch_size_must_follow_4n_plus_1() -> None:
    for bad in (1, 4, 7, 8, 10):
        with pytest.raises(PipelineError, match=r"4n\+1"):
            validate_seedvr2_batch_size(bad)
    for good in (5, 9, 13, 17):
        validate_seedvr2_batch_size(good)


def test_cli_rejects_seedvr2_batch_size_7_immediately(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["upscale", "work/missing", "--backend", "seedvr2", "--batch-size", "7"])
    captured = capsys.readouterr()
    assert code == 2
    assert "4n+1" in captured.err
    assert "5, 9, 13" in captured.err


def test_temporal_overlap_target_follows_schema_or_workflow() -> None:
    present = {
        "4": {"class_type": "TestUpscaler", "inputs": {"temporal_overlap": 0, "batch_size": 5}}
    }
    missing = {"4": {"class_type": "TestUpscaler", "inputs": {"batch_size": 5}}}
    settings = {"temporal_overlap": {"node_id": "4", "field": "temporal_overlap"}}
    schema = {
        "TestUpscaler": {"input": {"optional": {"temporal_overlap": ["INT", {"default": 0}]}}}
    }
    assert resolve_temporal_overlap_target(present, settings, {"TestUpscaler": {}}) == (
        "4",
        "temporal_overlap",
    )
    assert resolve_temporal_overlap_target(missing, settings, schema) == ("4", "temporal_overlap")
    assert resolve_temporal_overlap_target(missing, settings, {"TestUpscaler": {}}) is None


def test_temporal_overlap_stays_within_the_batch() -> None:
    validate_temporal_overlap(1, 5)
    validate_temporal_overlap(2, 9)
    validate_temporal_overlap(0, 5)
    with pytest.raises(PipelineError, match="temporal overlap"):
        validate_temporal_overlap(5, 5)
    with pytest.raises(PipelineError, match="temporal overlap"):
        validate_temporal_overlap(5, 9)


def test_seedvr2_batches_frames_and_refuses_per_frame_mode(tmp_path: Path) -> None:

    source = tmp_path / "hero.gif"
    frames = [character_frame(index, mark=(1, min(index + 1, 6))) for index in range(6)]
    save_rgba_gif(source, frames, [40] * 6)
    workdir = prepare_asset(
        source,
        scale=2,
        workdir=tmp_path / "job",
        alpha_mode="nearest",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    workflow = tmp_path / "workflow.json"
    config = tmp_path / "config.yaml"
    _workflow(workflow)
    _config(config, workflow)
    runner = DirectoryUpscaler(
        {"TestLoader": {}, "TestUpscaler": {}, "TestSaver": {}},
        scale=2,
    )
    from midnight_upscale.models import JobMetadata

    run_seedvr2(
        workdir,
        JobMetadata.load(workdir / "metadata.json"),
        batch_size=5,
        config_path=config,
        comfy_url=None,
        timeout_sec=None,
        overwrite=False,
        runner=runner,
    )
    assert len(runner.calls) == 2
    assert runner.calls[0]["4"]["inputs"]["batch_size"] == 5
    assert runner.calls[1]["4"]["inputs"]["batch_size"] == 5
    produced = list_indexed_frames(workdir / "upscaled_rgb")
    assert len(produced) == 6
    with Image.open(produced[0]) as first, Image.open(produced[5]) as last:
        assert first.size == (16, 16)
        assert first.getpixel((0, 0)) == (255, 0, 0)
        assert last.getpixel((0, 0)) == (255, 0, 0)
    assert "temporal_overlap" not in runner.calls[0]["4"]["inputs"]


def test_temporal_overlap_is_sent_when_the_node_exposes_it(tmp_path: Path) -> None:
    source = tmp_path / "hero.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(2, 2))], [40, 40])
    workdir = prepare_asset(
        source,
        scale=2,
        workdir=tmp_path / "job",
        alpha_mode="nearest",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    workflow = tmp_path / "workflow.json"
    config = tmp_path / "config.yaml"
    _workflow(workflow)
    _config(config, workflow)
    runner = DirectoryUpscaler(
        {
            "TestLoader": {},
            "TestUpscaler": {
                "input": {
                    "required": {"image": ["IMAGE"]},
                    "optional": {"temporal_overlap": ["INT", {"default": 0, "min": 0, "max": 16}]},
                }
            },
            "TestSaver": {},
        },
        scale=2,
    )
    from midnight_upscale.models import JobMetadata

    run_seedvr2(
        workdir,
        JobMetadata.load(workdir / "metadata.json"),
        batch_size=5,
        config_path=config,
        comfy_url=None,
        timeout_sec=None,
        overwrite=False,
        runner=runner,
        temporal_overlap=1,
    )
    assert runner.calls[0]["4"]["inputs"]["temporal_overlap"] == 1
    assert "alpha" not in Path(runner.calls[0]["1"]["inputs"]["directory"]).parts


def test_temporal_overlap_is_skipped_when_the_node_lacks_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "hero.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(2, 2))], [40, 40])
    workdir = prepare_asset(
        source,
        scale=2,
        workdir=tmp_path / "job",
        alpha_mode="nearest",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    workflow = tmp_path / "workflow.json"
    config = tmp_path / "config.yaml"
    _workflow(workflow)
    _config(config, workflow)
    runner = DirectoryUpscaler({"TestLoader": {}, "TestUpscaler": {}, "TestSaver": {}}, scale=2)
    from midnight_upscale.models import JobMetadata

    with caplog.at_level("WARNING"):
        run_seedvr2(
            workdir,
            JobMetadata.load(workdir / "metadata.json"),
            batch_size=5,
            config_path=config,
            comfy_url=None,
            timeout_sec=None,
            overwrite=False,
            runner=runner,
            temporal_overlap=1,
        )
    assert "temporal_overlap" not in runner.calls[0]["4"]["inputs"]
    assert "not an input" in caplog.text


def test_short_batch_does_not_invent_frames(tmp_path: Path) -> None:
    source = tmp_path / "hero.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(2, 2))], [40, 40])
    workdir = prepare_asset(
        source,
        scale=2,
        workdir=tmp_path / "job",
        alpha_mode="nearest",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    workflow = tmp_path / "workflow.json"
    config = tmp_path / "config.yaml"
    _workflow(workflow)
    _config(config, workflow)

    class ShortRunner(DirectoryUpscaler):
        def run_workflow(self, workflow: dict[str, dict[str, Any]]) -> dict[str, Any]:
            source_dir = Path(workflow["1"]["inputs"]["directory"])
            output_dir = Path(workflow["5"]["inputs"]["output_path"])
            frames = list_indexed_frames(source_dir)
            with Image.open(frames[0]) as image:
                image.save(output_dir / frames[0].name)
            return {"status": {"completed": True, "status_str": "success"}, "outputs": {}}

    from midnight_upscale.models import JobMetadata

    with pytest.raises(ValidationError, match="expected 2"):
        run_seedvr2(
            workdir,
            JobMetadata.load(workdir / "metadata.json"),
            batch_size=5,
            config_path=config,
            comfy_url=None,
            timeout_sec=None,
            overwrite=False,
            runner=ShortRunner({"TestLoader": {}, "TestUpscaler": {}, "TestSaver": {}}, scale=2),
        )
    assert list_indexed_frames(workdir / "upscaled_rgb") == []


def test_frame_upscale_does_not_pretend_to_upscale(tmp_path: Path) -> None:
    source = tmp_path / "hero.gif"
    save_rgba_gif(source, [character_frame(0), character_frame(1, mark=(2, 2))], [40, 80])
    workdir = prepare_asset(
        source,
        scale=2,
        workdir=tmp_path / "job",
        alpha_mode="lanczos",
        edge_cleanup="off",
        interpolate="none",
        overwrite=False,
    )
    from midnight_upscale.models import JobMetadata

    with pytest.raises(PipelineError, match="Real-ESRGAN is not bundled"):
        run_frame_upscale(
            workdir,
            JobMetadata.load(workdir / "metadata.json"),
            config_path=None,
            comfy_url=None,
            timeout_sec=None,
            overwrite=False,
            runner=None,
        )
    manifest = json.loads((workdir / "frame_upscale_manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 2
    assert manifest["target_width"] == 16
    assert list_indexed_frames(workdir / "upscaled_rgb") == []
    assert (workdir / "alpha" / "000000.png").is_file()


def _history_response(prompt_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            prompt_id: {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {},
            }
        },
        request=httpx.Request("GET", f"http://127.0.0.1:8188/history/{prompt_id}"),
    )


def test_socket_progress_uses_the_bus_saved_on_the_client() -> None:
    from midnight_upscale.progress import ProgressBus, bind_bus, current_bus, reset_bus

    client = ComfyClient("http://127.0.0.1:8188", timeout_sec=30, poll_interval_sec=0)
    bus = ProgressBus()
    token = bind_bus(bus)
    client._progress_bus = current_bus()
    reset_bus(token)
    assert current_bus() is None
    client._progress_bus.assign_prompt("prompt-a")
    client._progress_bus.set_nodes({"10": "KSampler"})
    client._progress_bus.apply_socket(
        json.dumps(
            {
                "type": "progress",
                "data": {"value": 1, "max": 1, "prompt_id": "prompt-a", "node": "10"},
            }
        )
    )
    event = bus.snapshot()
    assert event.node_step == 1
    assert event.node_steps_total == 1
    assert event.comfy_node_type == "KSampler"
    client.close()


def test_wait_keeps_polling_when_a_running_node_blocks_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ComfyClient("http://127.0.0.1:8188", timeout_sec=30, poll_interval_sec=0)
    calls = {"n": 0}

    def request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("timed out")
        return _history_response("pid")

    monkeypatch.setattr(client._client, "request", request)
    entry = client.wait("pid")
    assert calls["n"] == 3
    assert entry["status"]["completed"] is True
    client.close()


def test_wait_still_reports_a_dead_server_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ComfyClient("http://127.0.0.1:8188", timeout_sec=30, poll_interval_sec=0)

    def request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(client._client, "request", request)
    with pytest.raises(ComfyError, match="Could not reach ComfyUI"):
        client.wait("pid")
    client.close()


def test_wait_stops_when_comfy_stays_busy_past_the_job_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ComfyClient("http://127.0.0.1:8188", timeout_sec=0, poll_interval_sec=0)

    def request(method: str, url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(client._client, "request", request)
    with pytest.raises(ComfyError, match="Timed out after"):
        client.wait("pid")
    client.close()
