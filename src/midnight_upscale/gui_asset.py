"""Live PNG preparation using the converter's actual white-background remover."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from midnight_upscale.white_background import remove_white_background


def prepare_asset(source: str, warmth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Always adjust the original; preserve its dimensions and alpha exactly."""
    with Image.open(source) as image:
        if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Upload a single-frame PNG asset.")
        original = np.array(image.convert("RGBA"))
    adjusted = original.copy()
    amount = float(warmth)
    if not np.isfinite(amount) or not 0 <= amount <= 100:
        raise ValueError("Warmth must be between 0 and 100.")
    # Attenuate blue and, more gently, green: even clipped white can become warm.
    rgb = original[:, :, :3].astype(np.float32)
    rgb *= np.array([1.0, 1.0 - amount / 500, 1.0 - amount / 250])
    visible = original[:, :, 3] > 0
    adjusted[visible, :3] = np.rint(rgb[visible]).astype(np.uint8)
    removed = remove_white_background(adjusted)
    lost = visible & (removed[:, :, 3] == 0)
    faded = visible & (removed[:, :, 3] > 0) & (removed[:, :, 3] < adjusted[:, :, 3])
    marked = adjusted.copy()
    marked[lost] = [255, 0, 100, 255]
    marked[faded] = [255, 170, 0, 255]
    before = int(np.count_nonzero(visible & np.all(original[:, :, :3] == 255, axis=2)))
    after = int(np.count_nonzero(visible & np.all(adjusted[:, :, :3] == 255, axis=2)))
    lost_count, faded_count = int(lost.sum()), int(faded.sum())
    status = "No visible pixels affected." if not lost_count + faded_count else "Pixels at risk."
    report = (
        f"### {status}\n"
        f"Visible #FFFFFF pixels: **{before:,} → {after:,}**  \n"
        f"Fully removed (pink): **{lost_count:,}**  \n"
        f"Partially transparent (orange): **{faded_count:,}**  \n"
        f"Visible pixels: **{int(visible.sum()):,}**. Transparent pixels are excluded.\n\n"
        "Simulation uses VIDEO CONVERTER → White background, including edge cleanup. "
        "Animation generation, compression, and other removal modes can change the result."
    )
    return adjusted, marked, removed, report


def preview_asset(source: str | None, warmth: float):
    if not source:
        return None, None, None, "Upload a PNG to inspect it."
    import gradio as gr

    try:
        adjusted, marked, removed, report = prepare_asset(source, warmth)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    return Image.fromarray(adjusted), Image.fromarray(marked), Image.fromarray(removed), report


def export_asset(source: str | None, warmth: float) -> str:
    import gradio as gr

    if not source:
        raise gr.Error("Upload a PNG first.")
    try:
        adjusted, _, _, _ = prepare_asset(source, warmth)
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    destination = Path(tempfile.mkdtemp(prefix="midnight-asset-")) / "prepared_asset.png"
    Image.fromarray(adjusted).save(destination)
    return str(destination)


def mount_asset_tab() -> None:
    import gradio as gr

    gr.Markdown(
        "Prepare PNG assets before animation. Adjust warmth live and check which pixels "
        "White background removal would affect. Download preserves the original alpha and size."
    )
    source = gr.File(label="Input PNG", file_types=[".png"], type="filepath")
    warmth = gr.Slider(0, 100, value=0, step=1, label="Color warmth", info="0 = original colors")
    reset = gr.Button("Reset warmth")
    report = gr.Markdown("Upload a PNG to inspect it.")
    with gr.Row():
        adjusted = gr.Image(label="Prepared asset", image_mode="RGBA", interactive=False)
        marked = gr.Image(
            label="At risk: pink = removed, orange = faded", image_mode="RGBA", interactive=False
        )
        removed = gr.Image(
            label="White background removal simulation", image_mode="RGBA", interactive=False
        )
    save = gr.Button("Export prepared PNG", variant="primary")
    download = gr.File(label="Download prepared PNG", interactive=False)
    gr.on(
        triggers=[source.change, warmth.change],
        fn=preview_asset,
        inputs=[source, warmth],
        outputs=[adjusted, marked, removed, report],
        trigger_mode="always_last",
        concurrency_limit=1,
    )
    reset.click(lambda: 0, outputs=warmth)
    gr.on(triggers=[source.change, warmth.change], fn=lambda: None, outputs=download, queue=False)
    save.click(export_asset, inputs=[source, warmth], outputs=download)
