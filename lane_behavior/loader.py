"""Immutable admitted unmarked-behavior sequences, separate from legacy labels.

An independently produced admission report must bind every input file and every
replayed behavior record. This loader does not promote development captures or
substitute placeholder trajectory/control targets. The archive reader and CD210
camera ABI are reused unchanged; no old trainer is registered or modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from lane_dataset.config import sha256_file
from lane_dataset.loader import FrameArchiveReader, FRAME_DT
from lane_dataset.preprocessing import PACKED_FRAME_SHAPE
from lane_dataset.review_hold import assert_episode_not_held
from .records import SCHEMA as RECORD_SCHEMA, digest, training_corridor

SCHEMA = 'unmarked_behavior_sequence_v1'
ADMISSION_SCHEMA = 'unmarked_behavior_admission_v1'
REQUIRED_CHECKS = ('road_evidence_replayed','obstacle_evidence_replayed','physical_quality',
    'behavior_target_replayed','rendered_quality','split_integrity','retention_integrity','event_recovery')
LOGICAL_PASSING_BINDINGS = frozenset(('controlled_scene','rendered_free_space'))


def evidence_file_bindings(record):
    """Separate only the two typed passing claims from retained file digests."""
    bindings=record.get('evidence_bindings')
    if not isinstance(bindings,dict) or not bindings:raise ValueError('missing behavior evidence bindings')
    present=LOGICAL_PASSING_BINDINGS.intersection(bindings)
    permission=record.get('passing_space')
    if permission is None:
        if present:raise ValueError('logical passing claims require an observed permission')
    elif (present!=LOGICAL_PASSING_BINDINGS
            or bindings['controlled_scene']!=permission.get('controlled_scene_sha256')
            or bindings['rendered_free_space']!=permission.get('rendered_free_space_sha256')):
        raise ValueError('logical passing claims differ from observed permission')
    return {k:v for k,v in bindings.items() if k not in LOGICAL_PASSING_BINDINGS}


@dataclass(frozen=True)
class AdmittedBehaviorLabel:
    label_index: int
    record_sha256: str
    corridor: object


class AdmittedBehaviorSequence:
    """Load behavior-manifest.json only after independent production admission.

    Required manifest fields: schema, status='accepted', training_ready=True,
    capture_scope='production', split, accepted_label_count, frames_file,
    labels_file (NPZ containing only label_indices), behavior_records_file,
    admission_file, archives={narrow/wide:{path,sha256}}, files={path:sha256}.

    Admission requires schema=ADMISSION_SCHEMA, passed=True, method exactly
    'independent_raw_evidence_and_physical_replay', all REQUIRED_CHECKS True,
    input_files matching every inventory entry except the admission report, and
    validated_records in exact label order, each containing label_index,
    record_sha256, verified_evidence_bindings. Evidence names are shard-relative
    paths, except two explicitly replayed passing claims whose byte-bound proof
    must be retained and matched by passing_evidence_replay in the admission.
    rendered_quality denotes machine replay QA; it never implies human review.
    A development recording may be promoted by the independent publisher later;
    its final manifest and admission must satisfy this complete contract.
    """
    def __init__(self,manifest_path,*,corpus_root=None):
        self.path=Path(manifest_path).resolve()
        if self.path.is_dir():self.path=self.path/'behavior-manifest.json'
        if self.path.name!='behavior-manifest.json':
            raise ValueError('behavior loader requires its separate behavior-manifest.json')
        self.root=self.path.parent
        if corpus_root is None:
            if self.root.parent.name.casefold()!='accepted':
                raise ValueError('explicit corpus_root required outside corpus accepted directory')
            corpus_root=self.root.parent.parent
        self.corpus_root=Path(corpus_root).resolve()
        self._check_hold()
        self.manifest=json.loads(self.path.read_text(encoding='utf-8'))
        self.manifest_sha256=sha256_file(self.path)
        m=self.manifest
        if (m.get('schema')!=SCHEMA or m.get('status')!='accepted' or m.get('training_ready') is not True
                or m.get('capture_scope')!='production' or m.get('split') not in ('train','validation','test')):
            raise ValueError('requires independently admitted production unmarked behavior; development captures reject')
        self._verify_inventory()
        frames=[json.loads(row) for row in self._path(m['frames_file']).read_text(encoding='utf-8').splitlines() if row.strip()]
        self._validate_frames(frames)
        self.frames=tuple(MappingProxyType(f) for f in frames)
        self._frames_digest=digest(frames)
        with np.load(self._path(m['labels_file']),allow_pickle=False) as labels:
            if set(labels.files)!={'label_indices'}:
                raise ValueError('behavior labels contain only label_indices, without canonical control placeholders')
            indices=labels['label_indices'].copy()
        if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or not len(indices)
                or np.any(np.diff(indices.astype(np.int64))<=0) or indices[0]<0 or indices[-1]>=len(frames)
                or np.any(indices%10)):
            raise ValueError('behavior label indices must be ordered unique in-range 2 Hz indices')
        if m.get('accepted_label_count')!=len(indices):
            raise ValueError('accepted behavior count does not match label indices')
        if (self.timestamps[indices[0]]-self.timestamps[0]<5.-1e-6
                or self.timestamps[-1]-self.timestamps[indices[-1]]<10.-1e-6):
            raise ValueError('behavior sequences require five seconds history and ten seconds future')
        indices.flags.writeable=False
        self.labels=MappingProxyType(dict(label_indices=indices))
        self.records=json.loads(self._path(m['behavior_records_file']).read_text(encoding='utf-8'))
        if not isinstance(self.records,list) or len(self.records)!=len(indices):
            raise ValueError('one ordered behavior record per label index required')
        self.admission=json.loads(self._path(m['admission_file']).read_text(encoding='utf-8'))
        self._validate_admission()
        self.behavior_labels=tuple(self._load_label(int(index),record,proof)
            for index,record,proof in zip(indices,self.records,self.admission['validated_records']))
        self._records_digest=digest(self.records)
        self.archives=[]
        try:
            for camera in ('narrow','wide'):
                entry=m['archives'][camera]
                reader=FrameArchiveReader(self._path(entry['path']),expected_archive_sha256=entry['sha256'])
                self.archives.append(reader)
                if len(reader)!=len(frames) or reader.dtype!=np.uint8 or reader.frame_shape!=PACKED_FRAME_SHAPE:
                    raise ValueError('behavior packed archive count/dtype/shape mismatch')
        except Exception:
            self.close();raise

    def _path(self,relative):
        if not isinstance(relative,str) or not relative:
            raise ValueError('invalid behavior shard member')
        path=(self.root/relative).resolve()
        if self.root not in path.parents:
            raise ValueError('behavior shard member escapes manifest directory')
        return path

    def _check_hold(self):
        # Reuse the canonical episode identity even with our separate manifest.
        assert_episode_not_held(self.root/'manifest.json',corpus=self.corpus_root)

    def _live_guard(self):
        self._check_hold()
        if sha256_file(self.path)!=self.manifest_sha256:
            raise ValueError('behavior manifest changed after loading')

    def _verify_inventory(self):
        self._check_hold()
        inventory=self.manifest.get('files')
        if not isinstance(inventory,dict) or not inventory:
            raise ValueError('behavior immutable file inventory required')
        seen=set()
        for relative,expected in inventory.items():
            if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
                raise ValueError('invalid behavior SHA256 binding')
            path=self._path(relative)
            if path in seen or path==self.path:
                raise ValueError('duplicate or recursive behavior file binding')
            seen.add(path)
            if not path.is_file() or sha256_file(path)!=expected:
                raise ValueError(f'behavior file hash mismatch: {relative}')
        try:
            required=[self.manifest[key] for key in ('frames_file','labels_file','behavior_records_file','admission_file')]
            for camera in ('narrow','wide'):
                entry=self.manifest['archives'][camera];required.append(entry['path'])
                if inventory.get(entry['path'])!=entry['sha256']:
                    raise ValueError('behavior archive hash disagrees with file inventory')
        except (KeyError,TypeError) as error:
            raise ValueError('incomplete behavior file/archive declarations') from error
        if any(relative not in inventory for relative in required) or len(set(required))!=len(required):
            raise ValueError('behavior file inventory must distinctly bind all required artifacts')

    def _validate_frames(self,frames):
        if not frames:raise ValueError('behavior sequence requires frames')
        for index,f in enumerate(frames):
            if f.get('index')!=index or not isinstance(f.get('frame_id'),int) or isinstance(f['frame_id'],bool):
                raise ValueError('invalid behavior frame identity')
            for key,length in (('desire',8),('traffic_convention',2)):
                value=np.asarray(f.get(key),float)
                if value.shape!=(length,) or not np.isin(value,[0,1]).all() or value.sum()!=1:
                    raise ValueError(f'invalid behavior {key}')
        self.frame_ids=np.asarray([f['frame_id'] for f in frames],np.int64)
        self.timestamps=np.asarray([f['timestamp'] for f in frames],float)
        if (not np.isfinite(self.timestamps).all() or np.any(np.diff(self.frame_ids)!=1)
                or not np.allclose(np.diff(self.timestamps),FRAME_DT,rtol=0,atol=1e-6)):
            raise ValueError('behavior frames require uninterrupted synchronous 20 Hz')
        self.frame_ids.flags.writeable=False;self.timestamps.flags.writeable=False

    def _validate_admission(self):
        report=self.admission
        if (report.get('schema')!=ADMISSION_SCHEMA or report.get('passed') is not True
                or report.get('method')!='independent_raw_evidence_and_physical_replay'
                or any(report.get('checks',{}).get(key) is not True for key in REQUIRED_CHECKS)):
            raise ValueError('independent behavior admission checks have not all passed')
        expected={k:v for k,v in self.manifest['files'].items() if k!=self.manifest['admission_file']}
        if report.get('input_files')!=expected:
            raise ValueError('independent admission does not bind the exact immutable input inventory')
        rows=report.get('validated_records')
        if not isinstance(rows,list) or len(rows)!=len(self.records):
            raise ValueError('independent admission must validate every ordered behavior record')

    def _load_label(self,index,record,proof):
        if (record.get('schema')!=RECORD_SCHEMA or proof.get('label_index')!=index
                or proof.get('record_sha256')!=record.get('record_sha256')
                or record.get('label_frame_id')!=int(self.frame_ids[index])
                or abs(record.get('label_timestamp_s',float('inf'))-self.timestamps[index])>1e-6):
            raise ValueError('behavior record/frame/index mapping differs from independent admission')
        bindings=proof.get('verified_evidence_bindings')
        files=evidence_file_bindings(record)
        if (not isinstance(bindings,dict) or not bindings or bindings!=record.get('evidence_bindings')
                or any(self.manifest['files'].get(path)!=value for path,value in files.items())):
            raise ValueError('independently verified behavior evidence bindings do not match retained files')
        self._validate_passing_admission(record,proof,files)
        # Every causal evidence frame must actually be retained with its original
        # timestamp. A plausible ID/time in an unrelated proof is insufficient.
        evidence=[e for row in record['road']['evidence_frames'] for e in row]
        evidence.extend(e for o in record['obstacles'] for e in o['evidence_frames'])
        if record.get('passing_space') is not None:evidence.extend(record['passing_space']['evidence_frames'])
        by_id=dict(zip(self.frame_ids.tolist(),self.timestamps.tolist()))
        for e in evidence:
            timestamp=by_id.get(e['frame_id'])
            if (timestamp is None or abs(timestamp-e['timestamp_s'])>1e-6 or e['frame_id']>self.frame_ids[index]
                    or self.timestamps[index]-timestamp>5.+1e-6):
                raise ValueError('behavior evidence is future, stale, or missing from retained frames')
        corridor=training_corridor(record,verified_evidence_bindings=bindings)
        if corridor is None or not corridor.target.moving_supervision:
            raise ValueError('admitted behavior label has no valid lateral supervision')
        return AdmittedBehaviorLabel(index,record['record_sha256'],corridor)

    def _validate_passing_admission(self,record,admission,files):
        permission=record.get('passing_space');replay=admission.get('passing_evidence_replay')
        if permission is None:
            if replay is not None:raise ValueError('unrequested passing admission claim')
            return
        if not isinstance(replay,dict) or replay.get('method')!='independent_saved_free_surface_replay':
            raise ValueError('passing permission requires independent raw proof replay')
        relative=replay.get('proof_file')
        if relative not in files or files[relative]!=replay.get('proof_file_sha256'):
            raise ValueError('passing replay proof file is not byte-bound')
        saved=json.loads(self._path(relative).read_text(encoding='utf-8'))
        body=dict(saved);claimed=body.pop('proof_sha256',None)
        if (claimed!=digest(body) or digest(saved)!=replay.get('replayed_proof_document_sha256')
                or claimed!=permission['rendered_free_space_sha256']
                or saved.get('controlled_scene_sha256')!=permission['controlled_scene_sha256']
                or saved.get('supported') is not True):
            raise ValueError('retained passing proof differs from independent admission')
        from .passing import ObservedPassingSpace
        typed=ObservedPassingSpace.from_document(permission)
        projected={key:saved[key] for key in ('reference_sha256','allowed_intervals_m','opposing_width_m',
            'obstacle_ids','controlled_scene_sha256','evidence_frames')}
        projected['rendered_free_space_sha256']=claimed
        if ObservedPassingSpace.from_document(projected)!=typed:
            raise ValueError('retained free-space proof does not match record permission')

    def assert_admitted(self):
        self._live_guard();self._verify_inventory();self._validate_admission()
        if digest([dict(f) for f in self.frames])!=self._frames_digest or digest(self.records)!=self._records_digest:
            raise ValueError('behavior metadata or records changed in memory')
        if tuple(int(i) for i in self.labels['label_indices'])!=tuple(row.label_index for row in self.behavior_labels):
            raise ValueError('behavior labels changed after admission')
        for label in self.behavior_labels:label.corridor.verify_contract()

    def __len__(self):return len(self.frames)
    def frame(self,index):
        self._live_guard()
        return np.stack([reader[index] for reader in self.archives])
    def selected_targets(self):
        self._live_guard()
        return dict(label_indices=self.labels['label_indices'].copy())
    def close(self):
        for reader in getattr(self,'archives',[]):reader.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


BehaviorSequence=AdmittedBehaviorSequence
