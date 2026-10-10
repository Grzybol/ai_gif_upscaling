"""Local browser playback for exported animations, including spritesheet atlases."""

from __future__ import annotations

import json
from pathlib import Path

from midnight_upscale.video_export import APNG, GIF, PNG_SEQUENCE, SPRITESHEET, WEBM


def make_browser_player(
    outputs: dict[str, list[Path]],
    durations_ms: list[int],
    directory: Path,
    stem: str,
    *,
    video_preview: Path | None = None,
) -> Path:
    """Return a playable output, or write an HTML player for frame-based exports."""

    if SPRITESHEET in outputs:
        atlas_path = outputs[SPRITESHEET][-2]
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        images = [sheet["file"] for sheet in atlas["sheets"]]
        frames = [
            {
                "image": frame["sheet"],
                "x": frame["x"],
                "y": frame["y"],
                "w": frame["w"],
                "h": frame["h"],
                "duration": frame["duration_ms"],
            }
            for frame in atlas["frames"]
        ]
    elif PNG_SEQUENCE in outputs:
        frame_dir = outputs[PNG_SEQUENCE][0]
        images = [
            str(path.relative_to(directory)).replace("\\", "/")
            for path in sorted(frame_dir.glob("*.png"))
        ]
        if len(images) != len(durations_ms):
            raise ValueError("PNG sequence does not match the expected frame count")
        frames = [
            {"image": index, "x": 0, "y": 0, "w": 0, "h": 0, "duration": duration}
            for index, duration in enumerate(durations_ms)
        ]
    else:
        candidates = (
            outputs.get(APNG, [None])[0],
            outputs.get(GIF, [None])[0],
            video_preview,
            outputs.get(WEBM, [None])[0],
        )
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return candidate
        raise ValueError("No playable output was produced")

    destination = directory / f"{stem}_player.html"
    payload = json.dumps({"images": images, "frames": frames}).replace("<", "\\u003c")
    destination.write_text(_HTML.replace("__PAYLOAD__", payload), encoding="utf-8")
    return destination


_HTML = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Animation preview</title><style>
body{margin:24px;background:#202124;color:#eee;font:16px system-ui}
button,select,input{margin:8px;padding:8px}canvas{max-width:100%;max-height:80vh;
object-fit:contain;background-color:#bbb;background-image:conic-gradient(
#eee 25%,transparent 0 50%,#eee 0 75%,transparent 0);background-size:24px 24px}
</style><h1>Animation preview</h1><div>
<button id="play">Pause</button><label>Background <select id="bg">
<option value="checker">Checkerboard</option><option value="#000">Black</option>
<option value="#fff">White</option><option value="#606060">Gray</option></select></label>
<label>Speed <select id="speed"><option>0.25</option><option>0.5</option>
<option selected>1</option><option>2</option></select></label>
<input id="seek" aria-label="Frame" type="range" min="0" value="0">
<span id="status">Loading…</span></div><canvas id="canvas"></canvas><script>
const data=__PAYLOAD__;
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const play=document.getElementById('play'),seek=document.getElementById('seek');
const status=document.getElementById('status'),bg=document.getElementById('bg');
const speed=document.getElementById('speed');
let index=0,running=true,last=0,elapsed=0;
const images=data.images.map(src=>{const im=new Image();
im.src=src.split('/').map(encodeURIComponent).join('/');return im});
seek.max=data.frames.length-1;
function draw(){const f=data.frames[index],im=images[f.image];
const w=f.w||im.naturalWidth,h=f.h||im.naturalHeight;
if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
ctx.clearRect(0,0,w,h);ctx.drawImage(im,f.x,f.y,w,h,0,0,w,h);
seek.value=index;status.textContent=`Frame ${index+1} / ${data.frames.length}`;}
play.onclick=()=>{running=!running;play.textContent=running?'Pause':'Play';last=0};
seek.oninput=()=>{index=Number(seek.value);elapsed=0;draw()};
bg.onchange=()=>{canvas.style.backgroundImage=bg.value==='checker'?'':'none';
canvas.style.backgroundColor=bg.value==='checker'?'#bbb':bg.value};
function tick(now){if(running&&last){elapsed+=(now-last)*Number(speed.value);
const total=data.frames.reduce((s,f)=>s+Math.max(1,f.duration),0);elapsed%=total;
while(elapsed>=Math.max(1,data.frames[index].duration)){
elapsed-=Math.max(1,data.frames[index].duration);index=(index+1)%data.frames.length;}draw()}
last=now;requestAnimationFrame(tick)}
Promise.all(images.map(im=>im.decode())).then(()=>{draw();requestAnimationFrame(tick)})
.catch(()=>{status.textContent='Could not load animation';running=false});
document.addEventListener('visibilitychange',()=>{last=0});
</script></html>"""
