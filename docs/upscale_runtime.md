# Upscale runtime verification (RTX 4060 Ti 16 GB)

The Upscale tab was exercised against the running ComfyUI 0.28.3 instance on
`127.0.0.1:8188`. Its native SeedVR2 nodes and 3B INT8 model were available.
The installed `WavespeedFlashVSRNode` is a cloud API node, not a local
FlashVSR backend. No local FlashVSR weights or inference nodes were present.

| Run | Input | Output | Wall time | GPU evidence |
| --- | --- | --- | ---: | --- |
| FAST trial | frame 10 of `prop_plant.mp4`, 554×740 | 1108×1480 RGBA | 0.47 s | Anime4KCPP `-p cuda -d 0` detected RTX 4060 Ti |
| BALANCED trial | same frame | 1108×1480 RGBA | 4.12 s | `nvidia-smi`: up to 100% GPU, 10,490 MiB used |
| QUALITY trial | same frame | 1108×1480 RGBA | 9.88 s | `nvidia-smi`: up to 100% GPU, 12,373 MiB used |
| BALANCED batch | first five frames of the same MP4 | five frames, 1108×1480 VP9 WebM | 15.67 s | 97% GPU during a two-frame batch |
| QUALITY temporal | first five frames of the same MP4 | five 1108×1480 PNG frames | 24.05 s | 100% GPU, observed 13,755 MiB used; 5-frame window |
| QUALITY two windows | 10-frame 32×32 GIF | 10-frame 64×64 VP9 WebM | 7.17 s | completed frames reported in order |

The synthetic three-frame GIF used delays of 80/120/200 ms. Its 2× VP9
output had three 64×64 frames and a 0.400 s duration. Decoding that WebM
with `libvpx-vp9` recovered alpha values from 0 through 255. Its spritesheet
ZIP contained three 64×64 frames, the same delays, and a 400 ms duration in
JSON. The 2× MP4 retained three frames and a 0.400 s container duration;
decoding it recovered delays of 80/120/200 ms. QUALITY 4× produced three
128×128 MP4 frames with the same 0.400 s duration. A QUALITY PNG output
retained both opaque and transparent pixels.

Limits: the full 150-frame, five-second `prop_plant.mp4` was **not** run
end-to-end through QUALITY. The 16 GB VRAM policy uses five temporal frames
up to 2 million output pixels, and one-frame windows above that. At those
larger sizes QUALITY becomes spatial SeedVR2 rather than temporal-aware.
GPU and ComfyUI jobs have configurable timeouts, but completion time still
depends on source size and other workloads. MP4 cannot store alpha and is
composited against black; a silent audio stream holds its last frame for the
requested duration. PNG frame exports include `animation.json` for timing.
