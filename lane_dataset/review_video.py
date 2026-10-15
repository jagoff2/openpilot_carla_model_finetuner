"""Export continuous stock-crop clips from lossless archives for human review."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .config import sha256_file, write_json
from .preprocessing import DEPENDENCY_ROOT, MODEL_SIZE, reference_module
from .postprocess import shard_postprocess_lock


def review_annotations(manifest, telemetry, geometry):
    """Expose physical setup and measured lateral position in review-only text."""
    from .geometry import LaneReference
    reference=LaneReference(geometry['left_xyz'],geometry['right_xyz'],geometry['arc_m'])
    event=manifest.get('recovery_event') or {}
    onset=event.get('onset_index')
    held=event.get('held_offset_m')
    positions=np.asarray(telemetry['position_world'])
    progress=np.asarray(telemetry['route_s'])
    if len(positions)!=len(progress):raise ValueError('review telemetry count mismatch')
    annotations=[]
    for index,(position,route_s) in enumerate(zip(positions,progress)):
        _,offset=reference.project(position[:2],hint_s=float(route_s))
        side='right' if offset>=0 else 'left'
        position_text=f'CENTER OFFSET {offset:+.3f} m ({side})'
        if isinstance(onset,int) and held is not None:
            if index<onset:
                phase_text=f'RECOVERY SETUP: hold {float(held):+.2f} m | no labels'
            else:
                phase_text='CENTER RECOVERY: target 0.000 m'
        else:
            phase_text='CENTER TRACKING: target 0.000 m'
        annotations.append((position_text,phase_text))
    return annotations


def export_clip(shard):
    with shard_postprocess_lock(shard):
        return _export_clip_unlocked(shard)


def _export_clip_unlocked(shard):
    shard=Path(shard).resolve();manifest=json.loads((shard/"manifest.json").read_text())
    if manifest.get("status")=="accepted":
        raise ValueError("accepted shards are immutable; export review clips before publication")
    frames=[json.loads(line) for line in (shard/manifest["frames_file"]).read_text().splitlines()]
    with np.load(shard/'telemetry.npz',allow_pickle=False) as archive:
        telemetry={key:archive[key] for key in ('position_world','route_s')}
    annotations=review_annotations(manifest,telemetry,json.loads((shard/'geometry.json').read_text()))
    if len(annotations)!=len(frames):raise ValueError('review frame/telemetry count mismatch')
    try:
        import imageio_ffmpeg
        executable=imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        executable=shutil.which("ffmpeg")
    if executable is None: raise FileNotFoundError("ffmpeg is required for review clip export")
    source=DEPENDENCY_ROOT/"inspect_peripheral_cue.py"
    names={"unpack_six_plane_yuv420","packed_to_rgb_bt601_limited"}
    tree=ast.parse(source.read_text(encoding="utf-8-sig"));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={"np":np,"MODEL_SIZE":MODEL_SIZE}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),"exec"),ns)
    destination=shard/"review-stock-crops.mp4"
    if destination.exists():
        raise FileExistsError(destination)
    command=[executable,"-v","error","-f","rawvideo","-pixel_format","rgb24","-video_size","1024x296","-framerate","20","-i","-","-an","-c:v","libx264","-crf","20","-preset","fast","-pix_fmt","yuv420p","-movflags","+faststart",str(destination)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    readers=[]
    try:
        Reader=reference_module("frame_archive_v2").FrameArchiveReader
        for camera in ("narrow","wide"):
            entry=manifest["archives"][camera];readers.append(Reader(shard/entry["path"],expected_archive_sha256=entry["sha256"]))
        if any(len(reader)!=len(frames) for reader in readers):raise ValueError("review archive count mismatch")
        for index,frame in enumerate(frames):
            canvas=Image.new("RGB",(1024,296),(12,15,19));draw=ImageDraw.Draw(canvas)
            draw.text((10,5),f"STOCK MED | frame {frame['frame_id']} | {frame['timestamp']-frames[0]['timestamp']:.2f}s",fill="white")
            draw.text((524,5),"STOCK SBIG | development evidence, acceptance in manifest",fill="white")
            draw.text((10,22),annotations[index][0],fill='white')
            draw.text((524,22),annotations[index][1],fill=(255,210,100))
            for camera,reader in enumerate(readers):
                canvas.paste(Image.fromarray(ns["packed_to_rgb_bt601_limited"](reader[index])),(camera*512,40))
            process.stdin.write(np.asarray(canvas).tobytes())
        process.stdin.close();error=process.stderr.read().decode(errors="replace");code=process.wait()
        if code:raise RuntimeError(f"ffmpeg failed: {error}")
    finally:
        for reader in readers:reader.close()
        if process.poll() is None:process.terminate()
    manifest["review_clip"]=destination.name
    manifest.setdefault("files",{})[destination.name]=sha256_file(destination)
    write_json(shard/"manifest.json",manifest)
    return dict(path=str(destination),frames=len(frames),duration_s=len(frames)*.05,sha256=sha256_file(destination),reviewed=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("shard",type=Path)
    print(json.dumps(export_clip(parser.parse_args().shard),indent=2))
