"""Native SeedVR2 graph construction. These tests do not need a running ComfyUI."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from midnight_upscale.gui_logic import GuiJobConfig, preflight
from midnight_upscale.models import JobMetadata
from midnight_upscale.seedvr2_native import (
    assess_installation,
    build_native_workflow,
    choose_temporal_mode,
    format_comfy_check,
    padded_frame_count,
    run_seedvr2_native,
    select_unet,
    take_source_frames,
    transport_ffmpeg_command,
)
from midnight_upscale.utils import ValidationError, WorkflowConfigError


def _combo(choices: list[str]) -> list[object]:
    return [choices, {}]


def _image(name: str) -> dict[str, object]:
    return {"input": {"required": {name: ["IMAGE", {}]}}}


def _object_info(*, unet: list[str] | None = None, vae: list[str] | None = None) -> dict:
    unet_names = unet if unet is not None else ["seedvr2_3b_int8_convrot.safetensors"]
    vae_names = vae if vae is not None else ["seedvr2_ema_vae_fp16.safetensors"]
    return {
        "LoadVideo": {"input": {"required": {"file": _combo(["clip.mp4"])}}},
        "GetVideoComponents": {"input": {"required": {"video": ["VIDEO", {}]}}},
        "ImageScaleBy": {
            "input": {
                "required": {
                    "image": ["IMAGE", {}],
                    "upscale_method": _combo(["lanczos", "bilinear"]),
                    "scale_by": ["FLOAT", {"default": 1.0}],
                }
            }
        },
        "SeedVR2Preprocess": _image("resized_images"),
        "UNETLoader": {
            "input": {
                "required": {
                    "unet_name": _combo(unet_names),
                    "weight_dtype": _combo(["default", "fp8_e4m3fn"]),
                }
            }
        },
        "VAELoader": {"input": {"required": {"vae_name": _combo(vae_names)}}},
        "VAEEncodeTiled": {
            "input": {
                "required": {
                    "pixels": ["IMAGE", {}],
                    "vae": ["VAE", {}],
                    "tile_size": ["INT", {"default": 512}],
                    "overlap": ["INT", {"default": 64}],
                    "temporal_size": ["INT", {"default": 64}],
                    "temporal_overlap": ["INT", {"default": 8}],
                }
            }
        },
        "VAEDecodeTiled": {
            "input": {
                "required": {
                    "samples": ["LATENT", {}],
                    "vae": ["VAE", {}],
                    "tile_size": ["INT", {"default": 512}],
                    "overlap": ["INT", {"default": 64}],
                    "temporal_size": ["INT", {"default": 64}],
                    "temporal_overlap": ["INT", {"default": 8}],
                }
            }
        },
        "SeedVR2Conditioning": {
            "input": {"required": {"model": ["MODEL", {}], "vae_conditioning": ["LATENT", {}]}}
        },
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL", {}],
                    "seed": ["INT", {}],
                    "steps": ["INT", {"default": 20}],
                    "cfg": ["FLOAT", {"default": 8.0}],
                    "sampler_name": _combo(["euler", "dpmpp_2m"]),
                    "scheduler": _combo(["simple", "normal"]),
                    "positive": ["CONDITIONING", {}],
                    "negative": ["CONDITIONING", {}],
                    "latent_image": ["LATENT", {}],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
            }
        },
        "SeedVR2PostProcessing": {
            "input": {
                "required": {
                    "images": ["IMAGE", {}],
                    "original_resized_images": ["IMAGE", {}],
                    "color_correction_method": [
                        "COMBO",
                        {"options": ["lab", "wavelet", "adain", "none"], "default": "lab"},
                    ],
                }
            }
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE", {}],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            }
        },
        "SeedVR2TemporalChunk": {
            "input": {
                "required": {
                    "latent": ["LATENT", {}],
                    "temporal_overlap": ["INT", {"default": 0}],
                    "chunking_mode": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {"key": "auto", "inputs": {}},
                                {
                                    "key": "manual",
                                    "inputs": {
                                        "required": {"frames_per_chunk": ["INT", {"default": 21}]}
                                    },
                                },
                            ]
                        },
                    ],
                }
            }
        },
        "SeedVR2TemporalMerge": {
            "input": {"required": {"latents": ["LATENT", {}], "temporal_overlap": ["INT", {}]}}
        },
    }


def _classes(workflow: dict) -> set[str]:
    return {node["class_type"] for node in workflow.values()}


def test_prefers_3b_int8_and_does_not_select_7b() -> None:
    names = [
        "seedvr2_7b_int8_convrot.safetensors",
        "seedvr2_3b_fp8_e4m3fn.safetensors",
        "seedvr2_3b_int8_convrot.safetensors",
    ]
    assert select_unet(names) == "seedvr2_3b_int8_convrot.safetensors"
    assert select_unet(["seedvr2_7b_fp16.safetensors"]) is None
    assert select_unet(["qwen_image_edit_fp8_e4m3fn.safetensors"]) is None


def test_unchunked_workflow_uses_the_native_chain_and_official_sampler() -> None:
    workflow = build_native_workflow(
        _object_info(),
        video_name="rgb_transport.mp4",
        unet_name="seedvr2_3b_int8_convrot.safetensors",
        vae_name="seedvr2_ema_vae_fp16.safetensors",
        scale=2,
        temporal_mode="unchunked",
        temporal_overlap=1,
        seed=0,
    )
    classes = _classes(workflow)
    assert "SeedVR2TemporalChunk" not in classes
    assert "SeedVR2TemporalMerge" not in classes
    for name in (
        "LoadVideo",
        "SeedVR2Preprocess",
        "VAEEncodeTiled",
        "SeedVR2Conditioning",
        "KSampler",
        "VAEDecodeTiled",
        "SeedVR2PostProcessing",
        "SaveImage",
    ):
        assert name in classes
    sampler = next(node for node in workflow.values() if node["class_type"] == "KSampler")
    assert sampler["inputs"]["steps"] == 1
    assert sampler["inputs"]["cfg"] == 1.0
    assert sampler["inputs"]["sampler_name"] == "euler"
    assert sampler["inputs"]["scheduler"] == "simple"
    assert sampler["inputs"]["denoise"] == 1.0
    posted = next(
        node for node in workflow.values() if node["class_type"] == "SeedVR2PostProcessing"
    )
    assert posted["inputs"]["color_correction_method"] == "none"
    assert workflow["1"]["inputs"]["file"] == "rgb_transport.mp4"
    assert "alpha" not in str(workflow).lower()


def test_chunked_workflow_wires_overlap_from_the_chunk_node() -> None:
    workflow = build_native_workflow(
        _object_info(),
        video_name="rgb_transport.mp4",
        unet_name="seedvr2_3b_int8_convrot.safetensors",
        vae_name="seedvr2_ema_vae_fp16.safetensors",
        scale=2,
        temporal_mode="chunked",
        temporal_overlap=2,
        seed=0,
    )
    chunk = workflow["8"]
    merge = workflow["11"]
    assert chunk["class_type"] == "SeedVR2TemporalChunk"
    assert chunk["inputs"]["temporal_overlap"] == 2
    assert chunk["inputs"]["chunking_mode"] == "auto"
    assert "frames_per_chunk" not in chunk["inputs"]
    assert merge["class_type"] == "SeedVR2TemporalMerge"
    assert merge["inputs"]["temporal_overlap"] == ["8", 1]
    assert merge["inputs"]["latents"] == ["10", 0]
    encode = workflow["7"]
    assert encode["class_type"] == "VAEEncodeTiled"
    assert encode["inputs"]["tile_size"] == 512
    assert encode["inputs"]["pixels"] == ["4", 0]
    decode = workflow["12"]
    assert decode["class_type"] == "VAEDecodeTiled"
    assert decode["inputs"]["samples"] == ["11", 0]
    assert decode["inputs"]["tile_size"] == 512
    assert decode["inputs"]["overlap"] == 64
    assert decode["inputs"]["temporal_size"] == 64
    assert decode["inputs"]["temporal_overlap"] == 8


def test_manual_chunk_uses_the_dotted_dynamic_combo_key() -> None:
    workflow = build_native_workflow(
        _object_info(),
        video_name="rgb_transport.mp4",
        unet_name="seedvr2_3b_int8_convrot.safetensors",
        vae_name="seedvr2_ema_vae_fp16.safetensors",
        scale=2,
        temporal_mode="chunked",
        temporal_overlap=0,
        seed=0,
        frames_per_chunk=5,
    )
    chunk = workflow["8"]["inputs"]
    assert chunk["chunking_mode"] == "manual"
    assert chunk["chunking_mode.frames_per_chunk"] == 5
    assert "frames_per_chunk" not in chunk


def test_auto_keeps_a_short_clip_unchunked() -> None:
    assert (
        choose_temporal_mode(
            "auto", frame_count=9, width=512, height=768, scale=2, chunk_nodes=True
        )
        == "unchunked"
    )
    assert (
        choose_temporal_mode(
            "auto", frame_count=80, width=1024, height=1536, scale=2, chunk_nodes=True
        )
        == "chunked"
    )
    assert (
        choose_temporal_mode(
            "unchunked", frame_count=80, width=1024, height=1536, scale=2, chunk_nodes=True
        )
        == "unchunked"
    )


def test_padding_is_dropped_and_any_other_count_aborts() -> None:
    assert padded_frame_count(5) == 5
    assert padded_frame_count(6) == 9
    frames = [Path(f"{index}.png") for index in range(9)]
    assert take_source_frames(frames, 6) == frames[:6]
    with pytest.raises(ValidationError, match="aborted"):
        take_source_frames(frames[:7], 5)


def test_transport_command_is_lossless_and_constant_fps() -> None:
    command = transport_ffmpeg_command(
        "ffmpeg", Path("rgb/%06d.png"), 5, Path("out.mp4"), pix_fmt="yuv444p"
    )
    assert "24" in command
    assert command[command.index("-crf") + 1] == "0"
    assert "-b:v" not in command


def test_missing_models_are_reported_and_the_workflow_is_not_ready() -> None:
    report = assess_installation(
        _object_info(
            unet=["qwen_image_edit_fp8_e4m3fn.safetensors"],
            vae=["qwen_image_vae.safetensors"],
        ),
        {"devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4060 Ti : cudaMallocAsync"}]},
        ffmpeg_ok=True,
    )
    text = format_comfy_check(report)
    assert "ComfyUI: OK" in text
    assert "GPU: NVIDIA GeForce RTX 4060 Ti" in text
    assert "SeedVR2 native: OK" in text
    assert "Model: NOT FOUND" in text
    assert "seedvr2_3b_int8_convrot.safetensors" in text
    assert "Ready: NO" in text
    assert report.ready is False
    assert "7B" in "\n".join(report.missing)


def test_ready_report_names_the_3b_int8_model() -> None:
    report = assess_installation(_object_info(), ffmpeg_ok=True)
    text = format_comfy_check(report)
    assert "Model: seedvr2_3b_int8_convrot.safetensors" in text
    assert "VAE: seedvr2_ema_vae_fp16.safetensors" in text
    assert "Backend: seedvr2-native" in text
    assert "Workflow: OK" in text
    assert "Ready: YES" in text
    assert report.model_label == "3B INT8"


def test_rgba_frames_are_refused(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    Image.new("RGBA", (8, 8), (1, 2, 3, 4)).save(rgb / "000000.png")
    metadata = JobMetadata(
        source=str(tmp_path / "clip.gif"),
        original_width=8,
        original_height=8,
        frame_count=1,
        frame_durations_ms=[100],
        total_duration_ms=100,
        gif_loop_count=0,
        durations_constant=True,
        estimated_fps=10.0,
        has_transparency=True,
        has_semitransparency=False,
        requested_scale=2,
        target_width=16,
        target_height=16,
        alpha_mode="lanczos",
        edge_cleanup="off",
        interpolate="none",
    )
    metadata.save(tmp_path / "metadata.json")
    with pytest.raises(ValidationError, match="RGB"):
        run_seedvr2_native(
            tmp_path,
            metadata,
            comfy_url=None,
            config_path=None,
            timeout_sec=None,
            overwrite=True,
            temporal_overlap=0,
            client=object(),  # type: ignore[arg-type]
        )


def test_native_preflight_does_not_require_the_example_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("midnight_upscale.utils.shutil.which", lambda _name: "ffmpeg")
    source = tmp_path / "clip.gif"
    source.write_bytes(b"GIF89a")

    class Client:
        def object_info(self) -> dict[str, object]:
            return _object_info(
                unet=["wan2.2_ti2v_5B_fp16.safetensors"],
                vae=["wan2.2_vae.safetensors"],
            )

        def close(self) -> None:
            return None

    config = GuiJobConfig(
        source=source,
        backend="seedvr2-native",
        scale=2,
        batch_size=5,
        temporal_overlap=1,
        alpha_mode="lanczos",
        edge_cleanup="auto",
        output_format="webm",
        keep_workdir=False,
        comfy_url="http://127.0.0.1:8188",
        workflow_path="",
        output_dir=tmp_path,
        overwrite=False,
        verbose=False,
    )
    with pytest.raises(WorkflowConfigError, match="seedvr2_3b_int8_convrot") as caught:
        preflight(config, connect=lambda _url: Client())
    assert "REPLACE_" not in str(caught.value)
