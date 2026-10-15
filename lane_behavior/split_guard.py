"""Cross-pipeline geometry split protection for supplemental behavior shards.

Screening is advisory. Both publishers must recheck under the SAME
world_owner('corpus-publication', root) lock before publishing any accepted
manifest or changing the frozen split version. This module never writes data.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import errno
import json
import math
from pathlib import Path
import time

from lane_dataset.config import sha256_file
from lane_dataset.splits import GeometryGroup, GeometrySection, RoadInterval, SplitAssignment, audit_split_overlap
from lane_dataset.simulator import world_owner
from .loader import SCHEMA, ADMISSION_SCHEMA, REQUIRED_CHECKS


PUBLICATION_LOCK="corpus-publication"


@contextmanager
def publication_lock(root, *, timeout_s=600.):
    """Wait for normal concurrent publication, with a finite ten-minute bound."""
    if not math.isfinite(timeout_s) or not 0<timeout_s<=600.:
        raise ValueError("publication lock timeout must be positive and <=600 seconds")
    deadline=time.monotonic()+timeout_s
    with ExitStack() as stack:
        while True:
            try:
                stack.enter_context(world_owner(PUBLICATION_LOCK,root))
                break
            except OSError as error:
                if error.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK) and getattr(error,"winerror",None) not in (32,33):
                    raise
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError("corpus publication lock remained busy") from error
                time.sleep(min(.25,remaining))
        # Only acquisition errors retry. An exception from publication itself
        # propagates once, without accidentally executing the commit twice.
        yield


def geometry_section(value):
    if isinstance(value,GeometrySection):return value
    return GeometrySection(**{**value,"road_intervals":tuple(RoadInterval(**r) for r in value["road_intervals"])})


def _member(directory,name):
    if not isinstance(name,str) or not name:raise ValueError("invalid supplemental artifact path")
    path=(directory/name).resolve()
    if directory not in path.parents:raise ValueError("supplemental artifact escaped its shard")
    return path


def supplemental_sections(root):
    """Read exact admitted geometry metadata; invalid published metadata raises.

No large raw archive is rehashed during queue screening. The manifest binds
its split document and admission report; admission binds the same split file.
The actual publisher has already replayed all large physical/visual inputs.
"""
    result=[]
    for path in sorted((Path(root)/"accepted").glob("*/behavior-manifest.json")):
        path=path.resolve();directory=path.parent
        try:
            manifest=json.loads(path.read_text(encoding="utf-8"))
            if (manifest.get("schema")!=SCHEMA or manifest.get("status")!="accepted"
                    or manifest.get("training_ready") is not True or manifest.get("capture_scope")!="production"
                    or manifest.get("split") not in ("train","validation","test")
                    or type(manifest.get("accepted_label_count")) is not int or manifest["accepted_label_count"]<=0):
                raise ValueError("supplemental manifest is not a production admission")
            inventory=manifest["files"];split_path=directory/"behavior-split.json"
            split_sha=sha256_file(split_path)
            if inventory.get("behavior-split.json")!=split_sha:raise ValueError("supplemental split file hash changed")
            admission_path=_member(directory,manifest["admission_file"])
            if inventory.get(manifest["admission_file"])!=sha256_file(admission_path):
                raise ValueError("supplemental admission file hash changed")
            admission=json.loads(admission_path.read_text(encoding="utf-8"))
            if (admission.get("schema")!=ADMISSION_SCHEMA or admission.get("passed") is not True
                    or admission.get("method")!="independent_raw_evidence_and_physical_replay"
                    or any(admission.get("checks",{}).get(key) is not True for key in REQUIRED_CHECKS)
                    or admission.get("input_files",{}).get("behavior-split.json")!=split_sha):
                raise ValueError("supplemental split lacks independently bound admission")
            split=json.loads(split_path.read_text(encoding="utf-8"))
            if split.get("passed") is not True or split.get("split")!=manifest["split"]:
                raise ValueError("supplemental manifest and geometry split disagree")
            section=geometry_section(split["candidate_section"])
            result.append(dict(manifest=path,manifest_sha256=sha256_file(path),section=section,split=manifest["split"]))
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise ValueError(f"invalid supplemental split inventory {path}: {error}") from error
    return result


def supplemental_split_errors(root, *, section, split):
    """Return conflicts against all behavior publications, independent of IDs."""
    try:
        if split not in ("train","validation","test"):raise ValueError("explicit production split required")
        proposed=replace(geometry_section(section),section_id="pending-cross-pipeline-candidate")
        sections=[dict(section=proposed,split=split)]
        for index,entry in enumerate(supplemental_sections(root)):
            sections.append(dict(section=replace(entry["section"],section_id=f"supplemental-{index}"),split=entry["split"]))
        groups=[GeometryGroup(str(index),(item["section"],)) for index,item in enumerate(sections)]
        assignments=[SplitAssignment(str(index),item["split"],item["section"].family,1,0) for index,item in enumerate(sections)]
        return ["supplemental_split:"+error for error in audit_split_overlap(groups,assignments)]
    except (OSError,ValueError,KeyError,TypeError) as error:
        return ["supplemental_split_inventory_invalid:"+str(error)]


def assert_supplemental_split(root, *, section, split):
    """Call inside PUBLICATION_LOCK immediately before the publication commit."""
    errors=supplemental_split_errors(root,section=section,split=split)
    if errors:raise ValueError("; ".join(errors))
