"""Bounded read-only replay of selected saved curb frames."""
import json
from pathlib import Path

from lane_behavior.curb_evidence import measure_curb_road_frame, surveyed_reference
from lane_behavior.evidence import FrameStamp
from lane_dataset.config import write_json


def main():
    directory=Path(r'G:\lane-centering-dataset-v1\reports\unmarked\captures\20261015T131042-60299648')
    report=json.loads((directory/'capture-report.json').read_text())
    rows=json.loads((directory/'telemetry.json').read_text())
    frames={row['index']:row for row in map(json.loads,(directory/'evidence-frames.jsonl').read_text().splitlines())}
    bindings=report['retained_scene_bindings'];inventory=report['files']
    survey=directory/bindings['survey_replay']['path'];survey_sha=bindings['survey_replay']['sha256']
    geometry=surveyed_reference(survey,survey_sha,raw_survey_path=directory/bindings['raw_survey']['path'])
    results=[]
    for index in (250,350,450,550,650):
        frame=frames[index]
        station=geometry[0].project(rows[index]['position_world'][:2])[0]
        for camera,meta in frame['cameras'].items():
            proof=measure_curb_road_frame(payload=directory/meta['payload'],payload_sha256=meta['payload_sha256'],
                stamp=FrameStamp(frame['frame_id'],frame['timestamp_s'],camera),modality_stamps=meta['modality_stamps'],
                semantic_sensor_stamp=meta['separate_semantic_sensor_stamp'],depth_fixture=directory/'depth-fixture.json',
                depth_fixture_sha256=inventory['depth-fixture.json'],survey=survey,survey_sha256=survey_sha,
                current_s=station,surveyed_geometry=geometry)
            result=dict(index=index,camera=camera,station=station,**proof.audit)
            results.append(result)
            print(json.dumps({k:result[k] for k in ('index','camera','station','retained_by_side','rejections')}),flush=True)
    write_json(directory/'curb-dropout-diagnostic.json',results)


if __name__=='__main__':main()
