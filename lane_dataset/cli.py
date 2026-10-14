from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .config import Config, DEFAULT_ROOT, PROJECT, sha256_file, write_json
from .storage import Store


def prepare(root, inventory=False):
    config=Config();store=Store(root,config)
    config_path=Path(root)/"config.json"
    if config_path.exists():
        existing=json.loads(config_path.read_text())
        if existing.get("sha256")!=config.digest():
            raise RuntimeError("configuration differs from immutable prepared config")
    else:
        write_json(config_path,dict(config=config.document(),sha256=config.digest(),frozen=False))
    from .preprocessing import StockPreprocessor
    contract=StockPreprocessor().contract()
    write_json(Path(root)/"reports/preprocessing-contract.json",contract)
    model=Path(config.model_path)
    model_hash=sha256_file(model) if model.is_file() else None
    result=dict(status="development_not_qualified",storage=store.snapshot(),config_sha256=config.digest(),
                model=dict(path=str(model),sha256=model_hash,verified=model_hash==config.model_sha256),
                preprocessing=contract,accepted_counts={"train":0,"validation":0,"test":0},
                pending_gates=["steering_calibration","known_geometry_depth_validation","physical_controller_quality",
                               "native_and_procedural_geometry_review","one_and_two_worker_benchmark","frozen_disjoint_splits",
                               "crop_loss_coverage","per_geometry_clip_review","full_release_sequence_integration"])
    if inventory:
        from .inventory import inventory_maps
        result["inventory"]=inventory_maps(root)
    write_json(Path(root)/"reports/preparation.json",result)
    return result


def validate(root):
    from .splits import audit_release, GeometrySection, GeometryGroup, RoadInterval, SplitAssignment
    root=Path(root); store=Store(root)
    manifests=[];failures=[]
    for path in sorted((root/"accepted").glob("*/manifest.json")):
        try:
            doc=json.loads(path.read_text())
            for relative,digest in doc["files"].items():
                file=(path.parent/relative).resolve()
                if path.parent.resolve() not in file.parents or sha256_file(file)!=digest:
                    raise ValueError(f"hash mismatch: {relative}")
            from .loader import SequenceEpisode
            with SequenceEpisode(path.parent) as episode:
                episode.validate_all_frames()
            manifests.append(doc)
        except Exception as error:
            failures.append(dict(shard=str(path.parent),error=str(error)))
    try:
        groups=[];assignments=[]
        plan_path=root/"reports/scenario-plan.json"
        if plan_path.exists():
            plan=json.loads(plan_path.read_text())
            for group in plan["groups"]:
                sections=[]
                for section in group["sections"]:
                    section=dict(section);section["road_intervals"]=tuple(RoadInterval(**i) for i in section.get("road_intervals",[]))
                    sections.append(GeometrySection(**section))
                groups.append(GeometryGroup(group["group_id"],tuple(sections)))
            assignments=[SplitAssignment(**a) for a in plan["assignments"]]
        audit=audit_release([r for m in manifests for r in m.get("label_records",[])],groups,assignments)
    except Exception as error:
        audit=dict(qualified=False,reasons=[str(error)])
    result=dict(status="release_ready" if audit.get("qualified") and not failures else "incomplete",
                accepted_shards=len(manifests),audit=audit,validation_failures=failures,storage=store.snapshot())
    write_json(root/"reports/validation.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description="Stock MED/SBIG CARLA lane-centering corpus")
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare");p.add_argument("--inventory",action="store_true")
    p=sub.add_parser("calibrate");p.add_argument("--port",type=int,default=2500);p.add_argument("--extended",action="store_true")
    sub.add_parser("plan-scenarios")
    sub.add_parser("gallery")
    p=sub.add_parser("build-labels");p.add_argument("shard",type=Path)
    p=sub.add_parser("qualify-shard");p.add_argument("shard",type=Path)
    p=sub.add_parser("benchmark");p.add_argument("--port",type=int,default=2500);p.add_argument("--ticks",type=int,default=120);p.add_argument("--worker",type=int,default=1);p.add_argument("--speed",type=float,default=5.)
    p=sub.add_parser("generate");p.add_argument("--development",action="store_true");p.add_argument("--port",type=int,default=2500);p.add_argument("--ticks",type=int,default=500);p.add_argument("--worker",type=int,default=1);p.add_argument("--worker-count",type=int,choices=[1,2],default=2);p.add_argument("--speed",type=float,default=5.)
    for name in ("benchmark","generate"):
        sub.choices[name].add_argument("--traffic",type=int,choices=[0,2,3,4],default=0)
    sub.add_parser("validate");sub.add_parser("recover");sub.add_parser("export");sub.add_parser("status")
    args=parser.parse_args()
    if args.command=="prepare": result=prepare(args.root,args.inventory)
    elif args.command=="calibrate":
        if args.extended:
            from .calibrate_extended import capture, fit
            path=capture(args.root,args.port);result=fit(path)
            selected=result["calibrations_by_speed"].get("5.0")
            if selected and selected["qualified"]:write_json(path/"fit.json",selected)
            result["capture"]=str(path)
        else:
            from .calibration import capture_calibration
            from .controller import fit_steering_calibration
            import numpy as np
            path=capture_calibration(args.root,args.port);meta=json.loads((path/"manifest.json").read_text())
            with np.load(path/"samples.npz") as data:
                fitted=fit_steering_calibration(data["timestamp"][100:-1],data["command_steer"][101:],data["speed"][100:-1],data["angular_velocity_world"][100:-1,2],meta["wheelbase_m"],np.deg2rad(max(meta["wheel_steer_degrees"])))
            write_json(path/"fit.json",fitted.to_dict());result=dict(capture=str(path),fit=fitted.to_dict())
    elif args.command=="plan-scenarios":
        from .scenario_planner import plan_scenarios
        plan=plan_scenarios(args.root);result={k:plan[k] for k in ("status","selected_geometry_groups","episode_tasks","procedural_groups_by_split","deficits")}
    elif args.command=="gallery":
        from .gallery import build_gallery
        result=build_gallery(args.root)
    elif args.command in ("build-labels","qualify-shard"):
        from .admission import build_labels,qualify_candidate
        result=(build_labels if args.command=="build-labels" else qualify_candidate)(args.shard,dataset_root=args.root)
    elif args.command in ("benchmark","generate"):
        from .capture import capture_development, generate
        if args.command=="benchmark" or args.development:
            result=capture_development(args.root,args.port,args.worker,args.ticks,benchmark=args.command=="benchmark",task=dict(nominal_speed_mps=args.speed,traffic_count=args.traffic))
        else:
            result=generate(args.root,args.port,args.worker,args.worker_count)
    elif args.command=="validate": result=validate(args.root)
    elif args.command=="recover": result=dict(recovered=Store(args.root).recover())
    elif args.command=="status": result=dict(runs=Store(args.root).rows(),storage=Store(args.root).snapshot())
    elif args.command=="export":
        result=validate(args.root)
        if result["status"]!="release_ready":
            raise RuntimeError("release gates failed; no training export published")
        manifests=[str(p.relative_to(args.root)) for p in sorted((args.root/"accepted").glob("*/manifest.json"))]
        write_json(args.root/"training-index.json",dict(schema_version=1,episodes=manifests,config_sha256=Config().digest()))
    print(json.dumps(result,indent=2,default=str))
    # Development QA may remain unqualified, but an incomplete recording is a
    # process failure. The launcher must never confuse its JSON report with a
    # successful completed capture.
    if args.command in ("benchmark","generate") and result.get("manifest",{}).get("error"):
        raise SystemExit(2)


if __name__=="__main__":
    main()
