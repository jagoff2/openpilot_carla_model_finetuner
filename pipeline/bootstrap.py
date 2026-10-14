"""Prepare tests, measured calibration and native routes for unattended collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from lane_dataset.config import Config,DEFAULT_ROOT,PROJECT,sha256_file,write_json
from lane_dataset.settings import path_setting

ASSETS=Path(__file__).with_name('assets')


def doctor(root):
    import carla
    from lane_dataset.splits import APPROVED_TOWNS
    from lane_training.fetch_cd210 import EXPORTS,EXPORT_DIR,verify_file
    config=Config();installation=Path(config.carla_root)
    binary=installation/'CarlaUE4.exe'
    if not binary.exists():raise FileNotFoundError(f'Install CARLA 0.9.16 separately and set CARLA_ROOT: {binary}')
    client=carla.Client('127.0.0.1',2400)
    if client.get_client_version()!=config.carla_version:raise ValueError('CARLA client wheel and server must both be 0.9.16')
    for town in APPROVED_TOWNS:
        path=installation/f'CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr'
        if not path.is_file():raise FileNotFoundError(f'Required CARLA map is missing: {path}')
    for entry in EXPORTS.values():verify_file(EXPORT_DIR/entry['file'],entry)
    bundle=json.loads((ASSETS/'cd210-bundle.json').read_text())
    for role in bundle['models']:
        for kind in ('metadata',):
            entry=role[kind]
            if sha256_file(EXPORT_DIR/'bundle'/entry['fileName'])!=entry['downloadUri']['sha256']:
                raise ValueError('CD210 bundle files are missing or changed; rerun lane_training.fetch_cd210')
    from lane_dataset.cli import prepare
    prepare(root)
    result=dict(carla_version=config.carla_version,carla_root=str(installation),maps=list(APPROVED_TOWNS),
        model='CD210',dataset_root=str(root),checks_passed=True)
    write_json(root/'reports/setup-check.json',result)
    print(json.dumps(result),flush=True)


def tests(root):
    python=path_setting('training_python','LANE_TRAINING_PYTHON',PROJECT/'.venv-training/Scripts/python.exe')
    subprocess.run([str(python),'-m','pytest','tests','-q',f'--junitxml={root / "reports/tests.xml"}'],cwd=PROJECT,check=True)


def calibrate(root,port=2500):
    from reports.automatic_corpus import ensure_server
    from lane_dataset.capture import _calibration
    from lane_dataset.calibrate_extended import capture,fit
    missing=[]
    for speed in (1.5,2.5,3.5):
        try:_calibration(root,speed)
        except RuntimeError:missing.append(speed)
    if not missing:
        print(json.dumps(dict(stage='calibration_reused',speeds=[1.5,2.5,3.5])),flush=True);return
    ensure_server(root,port)
    directory=capture(root,port,missing,amplitudes=(.05,.1,.2,.25,.3,.35,.4))
    result=fit(directory)
    write_json(root/'reports/calibration-result.json',dict(capture=str(directory),**result))
    for speed in (1.5,2.5,3.5):_calibration(root,speed)


def seed_plan(root):
    import carla
    from lane_dataset.inventory import sample_route,route_document
    from lane_dataset.boundary_reference import lane_facing_marking_reference
    task=json.loads((ASSETS/'town06-seed.json').read_text());start=task['start']
    source=Path(Config().carla_root)/f'CarlaUE4/Content/Carla/Maps/OpenDrive/{task["town"]}.xodr'
    roadmap=carla.Map(task['town'],source.read_text())
    waypoint=roadmap.get_waypoint_xodr(start['road_id'],start['lane_id'],start['s'])
    if waypoint is None:raise ValueError('Town06 seed lane is absent from this CARLA installation')
    points=sample_route(waypoint,task.pop('route_length_m'),.5,task['branch'])
    _,route=route_document(points,task['town'])
    marks=[{side:dict(type=str(getattr(wp,side+'_lane_marking').type),width_m=float(getattr(wp,side+'_lane_marking').width))
        for side in ('left','right')} for wp in points]
    task['route']=lane_facing_marking_reference(route,marks)
    write_json(root/'planning/town06-flat-turn-plan.json',[task])


def plan(root):
    from lane_dataset.scenario_planner import PlannerConfig,plan_scenarios
    seed_plan(root)
    destination=root/'planning'
    if (destination/'reports/scenario-tasks.json').exists():
        print(json.dumps(dict(stage='plan_reused',path=str(destination))),flush=True);return
    result=plan_scenarios(destination,PlannerConfig(route_envelope_m=100.,minimum_episode_s=38.,
        start_spacing_m=5.,sample_step_m=.5,nominal_speed_mps=1.5,warmup_distance_budget_m=15.,planned_preview_m=25.))
    print(json.dumps({k:result[k] for k in ('status','selection','deficits')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--stage',choices=('all','doctor','tests','calibrate','plan'),default='all')
    parser.add_argument('--port',type=int,default=2500)
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    from reports.automatic_corpus import configure_root
    configure_root(root)
    if args.stage in ('all','doctor'):doctor(root)
    if args.stage in ('all','tests'):tests(root)
    if args.stage in ('all','calibrate'):calibrate(root,args.port)
    if args.stage in ('all','plan'):plan(root)
    print(json.dumps(dict(stage='preparation_finished',completed=args.stage,root=str(root))),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
