# Midnight Lounge upscaling

Local command-line pipeline for short transparent character loops. A GIF is the source format. The runtime asset is a transparent VP9 WebM. GIF preview output exists, and it is not the production format: a GIF only has one transparent color.

RGB and alpha are processed separately. SeedVR2 is a generative model. If it sees an RGBA frame it will repaint the coverage, and ComfyUI's own SeedVR2 preprocess drops alpha and fills the padding with black. Hair, dress edges, and the chair cutout need the original mask, only resampled. The model upscales color. A normal resampler upscales the mask. The two are combined afterward.

```
GIF
  -> composited RGBA frames, with GIF disposal applied
  -> RGB and alpha, not flattened onto a background
  -> SeedVR2 on RGB only
  -> Lanczos, bicubic, or nearest on alpha, with no threshold
  -> RGBA sequence
  -> VP9 WebM with alpha, using each frame's own delay
```

Nothing in the default path changes timing. `--interpolate` accepts only `none`.

## What you need

- Windows, Python 3.11 or newer
- An NVIDIA GPU for SeedVR2
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally, with its built-in SeedVR2 nodes
- FFmpeg, including `ffprobe`, on `PATH`
- SeedVR2 weights in the ComfyUI model folders: `seedvr2_3b_int8_convrot.safetensors` and `seedvr2_ema_vae_fp16.safetensors`

The numz [SeedVR2 custom nodes](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) are optional. `--backend seedvr2` still uses that pack and a mapped API workflow. `--backend auto` prefers the native nodes when they are installed.

FFmpeg is not installed by pip. In a new terminal after installing it:

```powershell
winget install Gyan.FFmpeg
ffmpeg -version
ffprobe -version
```

## Installation

From this repository:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Check the entry point:

```powershell
midnight-upscale --help
python -m midnight_upscale --help
```

## ComfyUI

Start ComfyUI so the API is on port 8188. `--highvram` keeps the 3B model on the GPU instead of streaming weights for every frame.

```powershell
python main.py --listen 127.0.0.1 --port 8188 --highvram
```

Confirm the install before a job:

```powershell
midnight-upscale comfy-check --comfy-url http://127.0.0.1:8188
```

`comfy-check` reports whether ComfyUI is reachable, the GPU, the native SeedVR2 nodes, the UNET and VAE the loaders can see, FFmpeg, and whether a native workflow can be built. `Ready: YES` means a native job can be queued. A missing model is listed by filename and the process exits with an error. It does not download weights and it does not install another SeedVR2 pack.

`comfy-info` still writes every installed node to JSON. Its numz summary (`SeedVR2VideoUpscaler`, `SeedVR2LoadDiTModel`, `SeedVR2LoadVAEModel`) only describes that custom-node pack.

Native jobs do not need `workflows/seedvr2.example.json`. That template is for `--backend seedvr2` only. Its `REPLACE_` class types are still rejected.

The native graph reads a temporary RGB video, upscales it, and writes PNG frames. The transport file is lossless H.264 (`crf 0`) at a constant 24 fps. That frame rate is only there so the container has a clock. The WebM still uses the original GIF delays from `metadata.json`. If SeedVR2 returns a different frame count, other than the extra tail frames its preprocess repeats to reach `4n+1`, the job aborts. Those repeated tail frames are dropped. Nothing is interpolated.

```powershell
copy config.example.yaml config.yaml
```

The numz backend still needs the workflow path and node ids in that file. `scale.mode: shortest_edge` writes `min(width, height) * scale` into `SeedVR2VideoUpscaler.resolution`, because that input is a pixel length, not a multiplier. Alpha is still scaled by the exact integer factor. If the model returns a different size, finalize stops instead of stretching the mask.

Keep `uniform_batch_size` false. Padding the last batch changes the frame count, and the job aborts.

ComfyUI has to be able to read and write the `work/` directory. Run it on the same machine as this tool.

## Inspect

```powershell
midnight-upscale inspect assets\wow.gif
midnight-upscale inspect assets\wow.gif --json
```

This reports size, frame count, every frame delay in milliseconds, total duration, average FPS, loop count, whether delays are constant or variable, and whether any pixel is transparent or semitransparent.

GIF disposal is applied while reading. A partial frame is not treated as a full picture. The source file is only opened for reading.

GIF transparency is one palette index. Semitransparent hair in the source is usually a hard mask plus fringe RGB stored under the transparent pixels. Lanczos on that mask produces grayscale edge values. Those values are kept. The mask is not forced back to 0 and 255.

## Prepare

```powershell
midnight-upscale prepare assets\wow.gif --scale 2
```

This creates `work/wow/`:

```
metadata.json
source_rgba/     composited RGBA PNGs
rgb/             straight RGB, not composited on black or white
alpha/           8-bit grayscale alpha
upscaled_alpha/  alpha resized by --scale
upscaled_rgb/    empty until upscale
final_rgba/
output/
```

`--alpha-mode` is `lanczos` (default), `bicubic`, or `nearest`.

`--edge-cleanup` is `auto` (default), `off`, or `simple`. `auto` uses `simple` when the GIF has any transparency and `off` when it does not. `simple` copies RGB from nearby opaque pixels into pixels whose alpha is 16 or less, and only within 3 source pixels of the silhouette. Alpha is not changed, so those pixels stay invisible. Pixels you can see (alpha above 16) keep their color. The point is to stop a black or wrong RGB value hidden in the GIF transparent index from becoming a dark fringe after Lanczos resizes the alpha mask. It does not fill the rest of the transparent canvas.

`--interpolate none` is the default and the only accepted value.

## Upscale

```powershell
midnight-upscale upscale work\wow --backend seedvr2 --comfy-url http://127.0.0.1:8188 --batch-size 5
```

`--batch-size` must be `4n+1` and at least 5: 5, 9, 13, and so on. The default is 5. `--batch-size 7` fails in this tool before ComfyUI is contacted. Frames are sent in consecutive groups of that size. The last group can be shorter. Frames are not duplicated to fill a group, and a failed group does not get a stand-in frame.

`--temporal-overlap` defaults to 1. It is written onto the upscaler only when that input already exists in the live node schema or in the workflow. If the installed node does not expose `temporal_overlap`, the run warns and continues, and the node keeps its own default (usually 0). Suggested values: batch 5 with overlap 1, batch 9 with overlap 1 or 2. Overlap stays at most 4 and smaller than the batch size.

`chunking` in `config.yaml` is `windows` or `all`. `windows` is the default and matches the command above. `all` submits every RGB frame in one prompt and still sets the node's own batch size from `--batch-size`, which is the better choice for a seamless loop when the GPU can hold the clip.

RGB is the only input. The alpha directory is never passed to ComfyUI.

### Fallback

```powershell
midnight-upscale upscale work\wow --backend frame-upscale
```

This writes `frame_upscale_manifest.json` and, unless `frame_upscale.workflow` points at an API-format image upscaler, stops. It does not run Real-ESRGAN, and it does not copy the original frames into `upscaled_rgb` and call that an upscale. Put the upscaled RGB PNGs in `upscaled_rgb` yourself, using the names `000000.png` onward, then run finalize.

Per-frame upscaling happens only when this backend is selected.

## Finalize

```powershell
midnight-upscale finalize work\wow --format webm
midnight-upscale finalize work\wow --format apng
midnight-upscale finalize work\wow --format gif --output output\wow_preview.gif
```

WebM is VP9 with alpha (`yuva420p` by default, or `--webm-pix-fmt yuva444p` for cleaner hair color). `-auto-alt-ref 0` is set because VP9 alpha breaks when alt-ref frames are enabled. The encoder reads an ffmpeg concat list that has one duration per frame. Variable GIF delays stay variable. They are not converted to a single FPS.

A GIF delay of 0 means "as fast as possible". Those frames are encoded at 10 ms. Every other delay is unchanged. The output duration is checked with ffprobe and the job fails if it drifts past a small tolerance.

WebM does not store a GIF loop count. Loop the file in the game runtime.

APNG keeps RGBA and maps the GIF loop count onto the APNG play count (0 means infinite).

`--format gif` is a preview. `palettegen` reserves a transparent color. Semitransparent pixels cannot survive in a GIF.

Before encoding, finalize checks that the decoded count, the upscaled RGB count, the upscaled alpha count, and the final RGBA count are the same, that every frame has the target size, and that sampled frames still have transparent and opaque alpha. If the source had semitransparent pixels, the output must still have some. A flattened or thresholded mask aborts the job.

## Full process

```powershell
python -m midnight_upscale process character.gif --scale 2 --comfy-url http://127.0.0.1:8188 --output output\character_upscaled.webm
```

The same command with the console script:

```powershell
midnight-upscale process assets\wow.gif --scale 2 --backend seedvr2 --batch-size 5 --format webm
```

Other flags: `--alpha-mode`, `--edge-cleanup auto|off|simple`, `--temporal-overlap`, `--interpolate none`, `--overwrite`, `--verbose`, `--keep-workdir` / `--no-keep-workdir`, `--config`, `--timeout`, `--crf`.

Work directories are kept unless you pass `--no-keep-workdir` after a successful `process`. A failed run keeps `work/` so you can fix the workflow and resume with `upscale` or `finalize`. Pass `--overwrite` to replace an existing job or output file.

A successful run prints:

```
SOURCE
1920x2560
120 frames
5.000 s

TARGET
3840x5120
120 frames
5.000 s

Output:
output/wow_upscaled.webm
```

## Web GUI

The page is a local frontend over the same prepare, upscale, and finalize functions the CLI uses. It does not replace the CLI.

```powershell
pip install -e ".[gui]"
```

Start ComfyUI, then:

```powershell
midnight-upscale-gui
```

`python -m midnight_upscale.gui` does the same thing. A browser opens to `http://127.0.0.1:7860`. The process listens on `127.0.0.1` only. There is no public share link. `--port 7861` changes the port. `--no-browser` skips opening a tab.

The decoder still accepts GIF only. PNG and animated WebP are refused with that reason. Several GIFs run one after another, never at the same time.

| Control | What it does |
| --- | --- |
| Backend | `Native ComfyUI` when those nodes are installed, `Numz custom node` for a mapped workflow, or the `frame-upscale` fallback. Check connection selects native when it is present. |
| Scale | 2x, 3x, or 4x. Alpha uses that integer. Native SeedVR2 scales by the same factor. The numz resolution knob is still the shortest edge in pixels. |
| Batch size | 5, 9, or 13. These apply to the numz node. They are `4n+1` and at least 5. |
| Temporal processing | Auto, Unchunked, or Chunked. Native only. Auto keeps a short clip unchunked. The job log says `Temporal mode: UNCHUNKED` or `CHUNKED`. |
| Temporal overlap | 0–3, default 1. Sent only when the installed node has that input. Chunked native runs log the value. |
| Alpha resize | Lanczos (default), Bicubic, or Nearest. Alpha never goes to SeedVR2. |
| Edge cleanup | Auto (default), Off, or Simple. Auto uses Simple when the GIF has transparency. |
| Output format | WebM Alpha (the production file), APNG, or GIF Preview. |
| Keep work directory | Off by default. A successful job then deletes `work/<name>`. Comparison frames are copied under `output/.gui/` first so the frame slider still works. |
| ComfyUI URL | Default `http://127.0.0.1:8188`. Check connection reads live `/object_info`. |
| Workflow JSON | Advanced. Only the numz backend reads it. Empty uses `config.yaml`. Native ComfyUI does not need an export. |
| Overwrite | Off. An existing output file stops the job. |
| Verbose | Tracebacks stay in the terminal. |
| Interpolation | None. RIFE is not implemented. |

`wow.gif` at 2x with SeedVR2 is written to `output/wow_2x_seedvr2.webm`. A WebM job can also write `wow_2x_seedvr2.browser-preview.mp4`. That MP4 is a checkerboard composite for the browser. It is not the transparent asset.

## Tests

Tests do not need ComfyUI or SeedVR2. The ComfyUI client is replaced with a fake that only checks batching and frame counts. Encoding tests run when `ffmpeg` is on `PATH`.

```powershell
pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
```

## Troubleshooting

**`ffmpeg` was not found.** Install it and open a new terminal. `winget install Gyan.FFmpeg`.

**Could not reach ComfyUI.** Start `python main.py --listen 127.0.0.1 --port 8188 --highvram`, or pass `--comfy-url`. `--highvram` keeps the 3B model on the GPU instead of streaming it for every frame.

**The example workflow was rejected.** That is intentional. `comfy-info`, then replace `workflows/seedvr2.example.json` with your API export. See [workflows/README.md](workflows/README.md).

**A class type is not available.** The prompt was not queued. The error lists the node id and class type. Match them to `comfy_object_info.json`.

**ComfyUI rejected the workflow.** The node error from `/prompt` is printed. A common cause is a loader field that is still named `directory` or `output_path` after you only renamed `class_type`. Use the field names from your export.

**The output folder is empty.** The saver did not write PNGs where `output.field` points, or it wrote more than one subdirectory of PNGs. Point the node at the directory this tool sets.

**Frame counts do not match.** A batch returned fewer or more PNGs than it was given. Nothing is dropped, repeated, or invented. `uniform_batch_size` and any "pad frames" option will cause this. Turn them off.

**Upscaled RGB size is not width times scale.** `SeedVR2VideoUpscaler.resolution` is the short side in pixels and can round. Alpha stays at the exact scale. Change `scale.mode`, or set the node so the output size matches `target_width` and `target_height` in `metadata.json`.

**Duration does not match.** A fixed `-r` was not used. If ffprobe still disagrees, the concat durations and the zero-delay warning in the log are the first place to look. Do not resample the GIF to a constant FPS to hide this.

**The WebM looks opaque or black in a player.** The player is not reading the alpha plane. `ffprobe` should show a `yuva` pixel format. Playback in Chrome needs a player that understands VP9 alpha. The PNG sequence in `final_rgba/` is the uncompressed check.

**Hair has a dark rim.** Transparent GIFs already use `simple` edge cleanup under `--edge-cleanup auto`. If the rim is actually visible, cleanup will not touch it. Try `--webm-pix-fmt yuva444p`. Do not threshold the alpha to fix it.

**`--batch-size 7` fails immediately.** SeedVR2 batch size must be `4n+1` and at least 5. Use 5, 9, 13, and so on.

**Alpha was thresholded in the GIF preview.** That format cannot store partial alpha. Use the WebM.

## Limitations

- The numz SeedVR2 graph is not automatic. Loader and saver node schemas are different on each machine, so they have to be mapped once in `config.yaml`. The native backend builds its graph from `/object_info` and does not use that file.
- `chunking: windows` does not share temporal context across the cut between batches. Use `chunking: all` when VRAM allows.
- A tail batch can be shorter than `--batch-size`. It is not padded.
- GIF delays are stored in hundredths of a second. A 45 ms delay in an editor is already 40 ms in the file. This tool keeps the file's delay.
- Zero delays become 10 ms at encode time.
- RIFE is not implemented. `--interpolate` cannot be anything but `none`.
- `--edge-cleanup simple` only recolors nearly invisible pixels next to the silhouette. It will not fix a fringe that is actually visible, and it does not change the alpha mask.
- WebM playback looping is the runtime's job.
