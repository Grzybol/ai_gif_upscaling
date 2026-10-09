# SeedVR2 workflow mapping

`seedvr2-native` builds the ComfyUI core graph from live `/object_info`. It does not use this file. The mapping below is only for `--backend seedvr2`, the numz `SeedVR2VideoUpscaler` pack.

The numz upscaler does not ship a runnable graph. Node inputs differ between that pack and ComfyUI's own SeedVR2 nodes. The tool loads the API-format JSON you give it, writes the paths and the scale, and refuses to queue the prompt if a class type is still a placeholder or is missing from `GET /object_info`.

RGB frames are the only images sent to ComfyUI. Alpha is resized locally.

## Verified SeedVR2 class types

These names were read from `numz/ComfyUI-SeedVR2_VideoUpscaler` (`node_id` inside `define_schema`) while this repo was written. `midnight-upscale comfy-info` reports whether your running ComfyUI actually has them. They are not submitted unless your workflow JSON contains them.

| Role | Class type | Inputs that matter here |
| --- | --- | --- |
| Upscaler | `SeedVR2VideoUpscaler` | `image` (an IMAGE batch), `dit`, `vae`, `resolution`, `batch_size`, `uniform_batch_size`, `temporal_overlap` (optional) |
| DiT loader | `SeedVR2LoadDiTModel` | `model`, `device` |
| VAE loader | `SeedVR2LoadVAEModel` | `model`, `device` |

`resolution` is the target length of the **shorter side** in pixels, and that node wants an even value. With `scale.mode: shortest_edge` this tool sets it to `min(width, height) * scale`, rounded up to the next even number when needed. Alpha is still resized to the exact `width * scale` by `height * scale`. Finalize stops if the RGB output size is different.

`batch_size` on `SeedVR2VideoUpscaler` must be `4n+1`. This tool requires at least 5 for video batches: 5, 9, 13, 17, and so on. `--batch-size 7` is rejected here, before ComfyUI sees the prompt.

`temporal_overlap` is optional on the current numz node (default 0, sensible values 1–4). `--temporal-overlap` (default 1) is written only when `/object_info` or the workflow already has that input. Suggested pairs: batch 5 with overlap 1, batch 9 with overlap 1 or 2.

`uniform_batch_size` must stay **false**. When it is true the node can pad the last batch, and this tool will abort because the PNG count no longer matches the source. Do not turn on any option that repeats frames into the saved sequence.

There is a second SeedVR2 implementation inside newer ComfyUI builds (`SeedVR2Preprocess` and related nodes). Its preprocess step drops alpha. Do not use that path to "restore" alpha. This pipeline recombines alpha itself after the RGB upscale.

## What you still have to map

The loader and the saver are not SeedVR2 nodes. Copy them from a workflow you have already run in your UI.

1. Install the SeedVR2 custom nodes and a node that can load every PNG in a directory. Video Helper Suite is a common choice. Confirm the real class type with `midnight-upscale comfy-info`; do not guess it.
2. In ComfyUI, build: directory of RGB PNGs → SeedVR2 DiT loader and VAE loader → `SeedVR2VideoUpscaler` → PNG sequence saver.
3. Connect only RGB. Do not connect the alpha mask, and do not use a node that joins alpha before the upscaler.
4. Turn on Dev mode and save the **API format** JSON.
5. Point `seedvr2.workflow` in `config.yaml` at that file.
6. Set `input`, `scale`, `batch_size`, and `output` node ids to the ids in that JSON. Set `field` to the real input name (`directory` in the example is only a placeholder name).
7. `output.mode: directory` means the saver writes PNGs into the folder this tool sets. Sorted file order must match frame order. `output.mode: history` downloads the images listed for that node instead.
8. Leave `model.node_id` empty if the exported workflow already selects the checkpoint you want.

`midnight-upscale upscale` fails before queueing if any `class_type` is missing from the live `/object_info` response. A placeholder `REPLACE_...` name fails even earlier, without contacting ComfyUI.

## Batching

`--batch-size 5` sends frames 0–4, then 5–9, and so on, when `chunking` is `windows`. The last window can be shorter. Frames are not duplicated to fill it, and a failed window does not get a substitute frame.

`chunking: all` sends the whole RGB sequence in one prompt and still writes `--batch-size` onto the upscaler node, which is how that node groups frames for temporal consistency. Use that for a seamless character loop when the GPU can hold the clip.

Per-frame image upscaling is `--backend frame-upscale` only.
