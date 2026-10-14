"""Offline gallery of actual captured audit frames and explicit review records.

Generating this gallery never marks a geometry reviewed or a shard accepted.
Review JSON must be checked against the bound manifest hash before use.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
from html import escape
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import quote

from .config import sha256_file


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _actual_file(manifest_path: Path, value: str, root: Path) -> Path | None:
    path = (manifest_path.parent / value).resolve()
    if not _inside(path, root):
        raise ValueError(f"audit path escapes dataset root: {value}")
    return path if path.is_file() else None


def _manifest_inventory(root: Path) -> list[tuple[Path, dict]]:
    result = []
    for bucket in ("accepted", "quarantine"):
        for path in sorted((root / bucket).glob("*/manifest.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"manifest must be an object: {path}")
            result.append((path, value))
    return result


def _representative_rank(item, root):
    path, manifest = item
    accepted = path.parent.parent.name == "accepted"
    ticks = manifest.get("measured_throughput", {}).get("ticks", 0)
    completed = not manifest.get("error") and isinstance(ticks, int) and ticks > 0
    clip_name = manifest.get("review_clip")
    clip = _actual_file(path, str(clip_name), root) if clip_name else None
    expected = manifest.get("files", {}).get(clip_name)
    bound_clip = bool(clip and expected and sha256_file(clip) == expected)
    # A completed continuous review is more useful than an older failed capture
    # with the same five audit timestamps. Qualification still stays pending.
    return (not accepted, not (completed and bound_clip), not completed,
            -len(manifest.get("audit_frames", [])), -ticks if completed else 0,
            -path.stat().st_mtime_ns, str(path))


def build_gallery(dataset_root: str | Path, output_path: str | Path | None = None, *,
                  max_embedded_bytes: int = 96 * 1024 * 1024) -> dict:
    """Render one representative episode for each captured geometry group.

    RGB audit frames and packed-crop previews are embedded, preserving a useful
    static artifact when the output is moved. Clips stay relative file links.
    A missing image/clip is reported rather than replaced with a synthetic one.
    """
    root = Path(dataset_root).resolve()
    output = Path(output_path).resolve() if output_path else root / "reports" / "review-gallery.html"
    if max_embedded_bytes <= 0:
        raise ValueError("gallery byte budget must be positive")
    grouped = defaultdict(list)
    for path, manifest in _manifest_inventory(root):
        metadata = manifest.get("metadata", {})
        group_id = manifest.get("geometry_group_id", manifest.get("group_id",
            metadata.get("geometry_group_id", metadata.get("group_id", f"unassigned:{path.parent.name}"))))
        grouped[str(group_id)].append((path, manifest))
    cards = []
    review_rows = []
    total_bytes = 0
    missing_paths = []
    image_count = 0
    for group_id, episodes in sorted(grouped.items()):
        episodes.sort(key=lambda item: _representative_rank(item, root))
        path, manifest = episodes[0]
        metadata = manifest.get("metadata", {})
        manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        state = "accepted" if path.parent.parent.name == "accepted" else "quarantine"
        family = manifest.get("family",metadata.get("family","unclassified"))
        split = manifest.get("split",metadata.get("split","development"))
        episode_id = manifest.get("episode_id",path.parent.name)
        rows = []
        audit_hashes = {}
        audit_frames = manifest.get("audit_frames", [])
        for audit in audit_frames:
            files = audit.get("files", {})
            if isinstance(files, list):
                files = {Path(name).stem:name for name in files}
            if not isinstance(files, dict):
                raise ValueError(f"audit frame files must be a mapping/list in {path}")
            figures = []
            for camera, relative_path in files.items():
                actual = _actual_file(path,str(relative_path),root)
                if actual is None:
                    missing_paths.append(str(path.parent / str(relative_path)))
                    figures.append(f'<p class="missing">Missing audit image: {escape(str(relative_path))}</p>')
                    continue
                mime = mimetypes.guess_type(actual.name)[0]
                if mime not in ("image/png","image/jpeg","image/webp"):
                    continue
                size = actual.stat().st_size
                if total_bytes + size > max_embedded_bytes:
                    raise ValueError("gallery image budget exceeded; increase explicitly or export smaller audit previews")
                total_bytes += size
                image_count += 1
                audit_hashes[str(actual.relative_to(path.parent)).replace(os.sep,"/")] = sha256_file(actual)
                uri = f"data:{mime};base64," + base64.b64encode(actual.read_bytes()).decode("ascii")
                figures.append(f'<figure><img loading="lazy" src="{uri}" alt="{escape(str(camera),quote=True)}">'
                               f'<figcaption>{escape(str(camera))}</figcaption></figure>')
            time_value = audit.get("timestamp_s",audit.get("timestamp",audit.get("time",audit.get("time_s","unknown"))))
            frame = audit.get("frame_id",audit.get("index","unknown"))
            stage = audit.get("stage","stage unannotated")
            rows.append(f'<div class="audit"><h4>Frame {escape(str(frame))} · t={escape(str(time_value))} s · '
                        f'{escape(str(stage))}</h4><div class="frames">{"".join(figures)}</div></div>')
        clip_value = manifest.get("review_clip")
        clip = _actual_file(path,str(clip_value),root) if clip_value else None
        clip_html = '<p class="missing">Continuous representative clip unavailable.</p>'
        if clip:
            clip_uri = quote(os.path.relpath(clip,output.parent).replace(os.sep,"/"),safe="/")
            clip_html = f'<video controls preload="metadata" src="{escape(clip_uri,quote=True)}"></video>'
        stages = sorted({str(audit.get("stage")) for audit in audit_frames if audit.get("stage")})
        review_rows.append({"geometry_group_id":group_id,"episode_id":episode_id,
            "manifest_relative_path":str(path.relative_to(root)).replace(os.sep,"/"),
            "manifest_sha256":manifest_sha,"status":"pending","reviewer":"","notes":"",
            "entry_reviewed":False,"apex_reviewed":False,"exit_reviewed":False,
            "crop_loss_intervals_reviewed":False,"continuous_clip_reviewed":False,
            "continuous_clip_available":bool(clip),"audit_stages_available":stages})
        review_rows[-1]["audit_file_sha256"] = audit_hashes
        review_rows[-1]["review_clip_sha256"] = sha256_file(clip) if clip else None
        item_index = len(review_rows)-1
        cards.append(f'<article data-index="{item_index}"><header><h2>{escape(group_id)}</h2>'
            f'<p>{escape(str(family))} · {escape(str(split))} · <b>{state}</b> · {len(audit_frames)} audit timestamps</p>'
            f'<p>Episode {escape(str(episode_id))}; {len(episodes)} appearance/capture variant(s).</p></header>'
            f'{clip_html}<details open><summary>Captured audit frames</summary>{"".join(rows) if rows else "<p>No audit frames recorded.</p>"}</details>'
            '<fieldset><legend>Review disposition</legend><label>Status <select class="status">'
            '<option value="pending">Pending</option><option value="pass">Pass</option><option value="reject">Reject</option></select></label>'
            '<label>Reviewer <input class="reviewer" type="text"></label>'
            + ''.join(f'<label><input type="checkbox" class="{name}" {"disabled" if name == "continuous_clip_reviewed" and not clip else ""}> {label}</label>'
                for name,label in [("entry_reviewed","Entry"),("apex_reviewed","Apex"),("exit_reviewed","Exit"),
                    ("crop_loss_intervals_reviewed","Crop-loss intervals"),("continuous_clip_reviewed","Continuous clip")])
            + '<label>Notes <textarea class="notes" rows="2"></textarea></label></fieldset></article>')
    data = json.dumps({"schema_version":1,"review_origin":"explicit_human_disposition",
                       "geometry_reviews":review_rows},ensure_ascii=True).replace("<","\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lane-centering corpus visual review</title><style>
body{font:15px system-ui,sans-serif;color:#e8edf4;background:#101923;margin:0;padding:24px;max-width:1680px;margin-inline:auto}
h1{font-size:28px}h2{font-size:20px}h4{margin:10px 0}.notice,article{background:#182635;border:1px solid #34506b;border-radius:10px;padding:18px;margin:18px 0}
.notice{border-left:5px solid #eeb860}.frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:10px}
figure{margin:0;background:#081018;padding:6px}img{width:100%;height:auto;display:block;image-rendering:auto}figcaption{text-align:center;padding:5px;color:#c1d3e5}
video{width:100%;max-height:720px}.audit{border-top:1px solid #34495e;padding:10px 0}.missing{color:#ffcf8c}
fieldset{display:flex;gap:16px;flex-wrap:wrap;border:1px solid #47647d;margin-top:16px}label{display:block}textarea{display:block;min-width:320px}input,select,textarea,button{font:inherit;padding:6px}button{cursor:pointer;background:#c6edff;border:0;border-radius:5px;padding:10px 14px}
summary{cursor:pointer;padding:10px 0}.summary{color:#b6c9db}code{word-break:break-word}
</style><h1>Lane-centering corpus visual review</h1>
<p class="notice">Only recorded audit imagery appears here. Creating or opening this gallery does not approve a geometry group.
Every disposition starts pending. A passing release review requires entry, apex, exit, crop-loss intervals and a continuous representative clip.</p>
''' + f'<p class="summary">{len(grouped)} captured geometry groups · {image_count} embedded images · {len(missing_paths)} missing audit files</p>' + \
        '<button id="download">Download review dispositions</button>' + ''.join(cards) + \
        '<script type="application/json" id="review-data">'+data+'</script>' + '''<script>
document.getElementById('download').addEventListener('click',()=>{
const result=JSON.parse(document.getElementById('review-data').textContent);
for(const card of document.querySelectorAll('article')){
const row=result.geometry_reviews[Number(card.dataset.index)];
for(const name of ['status','reviewer','notes'])row[name]=card.querySelector('.'+name).value;
for(const name of ['entry_reviewed','apex_reviewed','exit_reviewed','crop_loss_intervals_reviewed','continuous_clip_reviewed'])row[name]=card.querySelector('.'+name).checked;
if(row.status==='pass'&&(!row.reviewer.trim()||!row.entry_reviewed||!row.apex_reviewed||!row.exit_reviewed||!row.crop_loss_intervals_reviewed||!row.continuous_clip_reviewed)){
alert('Complete reviewer identity and every coverage checkbox for '+row.geometry_group_id+' before marking pass.');return;}}
result.reviewed_at_utc=new Date().toISOString();const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='geometry-review-dispositions.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
});</script></html>'''
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(page,encoding="utf-8")
    template = output.with_suffix(".review-template.json")
    template.write_text(json.dumps(json.loads(data),indent=2),encoding="utf-8")
    return {"gallery_path":str(output),"review_template_path":str(template),
            "geometry_groups":len(grouped),"embedded_images":image_count,"embedded_bytes":total_bytes,
            "missing_audit_files":missing_paths,"reviewed_geometry_groups":0,
            "review_status":"pending_explicit_dispositions"}


def validate_review_dispositions(dataset_root: str | Path, review_file: str | Path) -> dict:
    """Bind explicit review decisions to unchanged, existing capture manifests."""
    root = Path(dataset_root).resolve()
    review = json.loads(Path(review_file).read_text(encoding="utf-8"))
    errors = []
    passed = []
    seen = set()
    if review.get("review_origin") != "explicit_human_disposition":
        errors.append("missing_explicit_review_origin")
    for row in review.get("geometry_reviews",[]):
        group = row.get("geometry_group_id")
        if not group or group in seen:
            errors.append("missing_or_duplicate_review_group")
            continue
        seen.add(group)
        if row.get("status") != "pass":
            errors.append(f"review_not_passed:{group}")
            continue
        path = (root / row.get("manifest_relative_path","")).resolve()
        if not _inside(path,root) or not path.is_file():
            errors.append(f"invalid_review_manifest_path:{group}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("manifest_sha256"):
            errors.append(f"review_manifest_hash_mismatch:{group}")
        if not str(row.get("reviewer","")).strip():
            errors.append(f"missing_reviewer:{group}")
        required = ("entry_reviewed","apex_reviewed","exit_reviewed",
                    "crop_loss_intervals_reviewed","continuous_clip_reviewed")
        if not all(row.get(name) is True for name in required):
            errors.append(f"incomplete_review_coverage:{group}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        metadata = manifest.get("metadata",{})
        manifest_group = manifest.get("geometry_group_id",manifest.get("group_id",
            metadata.get("geometry_group_id",metadata.get("group_id"))))
        if manifest_group != group:
            errors.append(f"review_geometry_identity_mismatch:{group}")
        clip_value = manifest.get("review_clip")
        clip = _actual_file(path,str(clip_value),root) if clip_value else None
        if not clip:
            errors.append(f"missing_continuous_review_clip:{group}")
        elif sha256_file(clip)!=row.get("review_clip_sha256"):
            errors.append(f"review_clip_hash_mismatch:{group}")
        audit_hashes=row.get("audit_file_sha256",{})
        if not audit_hashes:
            errors.append(f"missing_review_audit_hashes:{group}")
        for relative,digest in audit_hashes.items():
            try:
                actual=_actual_file(path,relative,root)
            except ValueError:
                actual=None
            if actual is None or sha256_file(actual)!=digest:
                errors.append(f"review_audit_hash_mismatch:{group}")
        if not any(error.endswith(f":{group}") for error in errors):
            passed.append(group)
    if not seen:
        errors.append("no_geometry_reviews")
    return {"qualified":not errors,"errors":errors,"passed_geometry_groups":passed}
