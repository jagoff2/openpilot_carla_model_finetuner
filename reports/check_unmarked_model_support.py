"""Read-only stock CD210 inference on candidate footage, never admission/training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lane_dataset.loader import FrameArchiveReader
from lane_dataset.config import sha256_file,write_json
from lane_training.cd210 import student_rollout,ROLLOUT_PLAN
from lane_training.cd210_model import create_model
from lane_training.smoke import parameter_digest
from lane_behavior.records import training_corridor
from lane_behavior.loss import model_behavior_loss


class CandidateFrames:
    def __init__(self,path,records):
        self.path=path
        self.report=json.loads((path/'capture-report.json').read_text())
        self.frames=[json.loads(v) for v in (path/'frames.jsonl').read_text().splitlines()]
        self.labels={'label_indices':np.array([r['label_index'] for r in records])}
        assert sha256_file(path/'frames.jsonl')==self.report['files']['frames.jsonl']
        self.archives=[FrameArchiveReader(path/self.report['archives'][c]['path'],
            expected_archive_sha256=self.report['archives'][c]['sha256']) for c in ('narrow','wide')]
    def __len__(self):return len(self.frames)
    def frame(self,index):return np.stack([v[index] for v in self.archives])
    def selected_targets(self):return self.labels
    def close(self):
        for v in self.archives:v.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('records',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=json.loads(args.records.read_text());corridors=[]
    for row in records:
        record=row['record']
        for relative,expected in record['evidence_bindings'].items():
            assert sha256_file(args.capture/relative)==expected
        corridors.append(training_corridor(record,verified_evidence_bindings=record['evidence_bindings']))
    torch.set_num_threads(1)
    model=create_model().eval().requires_grad_(False)
    before=parameter_digest(model);count=0
    @torch.no_grad()
    def forward(inputs):
        nonlocal count
        result=model(inputs);count+=1
        if count%100==0:print(json.dumps(dict(event='read_only_stock_support',frame=count)),flush=True)
        return result.detach()
    episode=CandidateFrames(args.capture,records)
    try:prediction,_=student_rollout(episode,forward)
    finally:episode.close()
    with torch.no_grad():loss,report=model_behavior_loss(prediction,prediction,corridors,adapter=ROLLOUT_PLAN)
    assert before==parameter_digest(model)
    assert all(p.grad is None for p in model.parameters())
    result=dict(scope='read_only_stock_support_on_unadmitted_capture',training_performed=False,admission_granted=False,
        capture=str(args.capture),capture_report_sha256=sha256_file(args.capture/'capture-report.json'),
        records_sha256=sha256_file(args.records),parameter_sha256=before,candidate_labels=len(records),
        diagnostic_loss=float(loss),diagnostics=report)
    write_json(args.output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
