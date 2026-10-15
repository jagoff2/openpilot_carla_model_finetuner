"""Isolated CD210 mixed-objective steps and read-only data qualification.

This module never changes the source-bound legacy trainer. A mixed step evaluates
both losses at the same parameters, combines their normalized gradients with an
explicit recipe, and leaves the optimizer decision to the caller. Test data is
audited for split leakage but never loaded into training or validation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from lane_dataset.config import sha256_file
from lane_dataset.loader import SequenceEpisode
from lane_dataset.review_hold import active_holds, assert_episode_not_held
from lane_training.cd210 import ROLLOUT_PLAN, backward as legacy_backward, student_rollout
from lane_training.centerline import OBJECTIVE_VERSION as LEGACY_OBJECTIVE, LossConfig
from lane_training.checkpoint_store import lateral_score, can_promote
from lane_training.production import discover as discover_legacy, evaluate_episode
from lane_training.smoke import parameter_digest
from .loader import AdmittedBehaviorSequence
from .loss import OBJECTIVE_VERSION as BEHAVIOR_OBJECTIVE, model_behavior_loss
from .training import behavior_backward


OBJECTIVE_VERSION = 'cd210_legacy_and_admitted_behavior_mixture_v1'
SELECTION_METRIC = 'separate_legacy_and_behavior_lateral_nonregression_v1'
KINDS = ('legacy', 'behavior')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Recipe:
    """Per-objective weights, independent of each cohort's episode/label count."""
    legacy_weight: float = .5
    behavior_weight: float = .5
    vision_lr: float = 1e-6
    policy_lr: float = 1e-5
    weight_decay: float = .01
    gradient_clip: float = 1.
    cpu_threads: int = 2
    device: str = 'cpu'

    def __post_init__(self):
        positive = (self.legacy_weight, self.behavior_weight, self.vision_lr,
                    self.policy_lr, self.gradient_clip)
        if (not all(math.isfinite(v) and v > 0 for v in positive)
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not isinstance(self.cpu_threads, int) or self.cpu_threads < 1):
            raise ValueError('invalid mixed recipe hyperparameters')
        if not math.isclose(self.legacy_weight+self.behavior_weight, 1., abs_tol=1e-12):
            raise ValueError('explicit objective weights must sum to one')
        if self.device != 'cpu':
            raise ValueError('mixed extension reserves both GPUs for collection; device must be cpu')

    def document(self):
        return dict(**asdict(self), objective_version=OBJECTIVE_VERSION,
            legacy_objective=LEGACY_OBJECTIVE, behavior_objective=BEHAVIOR_OBJECTIVE,
            normalization='each episode mean over fixed supported labels, then explicit objective weights',
            selection_metric=SELECTION_METRIC)


@dataclass(frozen=True)
class EpisodeRef:
    kind: str
    path: str
    sha256: str
    split: str
    group: str
    family: str
    labels: int

    @property
    def key(self):
        return self.kind+':'+self.sha256

    def lineage(self):
        # Existing review-hold parsing recognizes manifest.json, not the new
        # behavior filename. Bind the canonical episode identity explicitly.
        return dict(**asdict(self), id=str(Path(self.path).parent/'manifest.json'))


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def refresh_behavior_split(root, task, task_sha, saved, split):
    """Preserve original admission, recheck additions under the publisher lock.

    Legacy collection appends frozen geometry versions while this reader runs.
    A newer compatible baseline changes the audit used for discovery, never the
    immutable behavior manifest, original assignment, or training identity.
    """
    from .admission import verify_split,current_split_assignment
    from .split_guard import publication_lock
    with publication_lock(root):
        assignment=saved.get('assignment')
        if (saved.get('passed') is not True or saved.get('split')!=split or not isinstance(assignment,dict)
                or assignment.get('candidate_task_sha256')!=task_sha or assignment.get('split')!=split
                or assignment.get('frozen_split_sha256')!=saved.get('baseline_sha256')):
            raise ValueError('original behavior admission exact-task/split binding changed')
        original_path=Path(saved['baseline_path'])
        if not original_path.is_absolute():original_path=Path(root)/original_path
        if sha256_file(original_path)!=saved['baseline_sha256']:
            raise ValueError('original behavior admission baseline changed')
        original=_read(original_path)
        if fingerprint(original)!=fingerprint(saved['baseline_document']):
            raise ValueError('original behavior admission baseline document changed')
        current=current_split_assignment(root,assignment)
        audit=verify_split(root,task,task_sha,current)
        if (audit['passed'] is not True or audit['split']!=split
                or fingerprint(audit['candidate_section'])!=fingerprint(saved.get('candidate_section'))):
            raise ValueError('behavior split replay failed or changed: '+str(audit.get('errors')))
        # An append-only publication must retain earlier geometry and assignment
        # identities exactly; a removal/reassignment cannot silently bless data.
        for name in ('groups','assignments'):
            retained={fingerprint(value) for value in audit['baseline_document'][name]}
            if any(fingerprint(value) not in retained for value in original[name]):
                raise ValueError('current baseline rewrote previously admitted split geometry')
        return audit


def discover(corpus, *, verify_payloads=True):
    """Combine the published legacy index and independently admitted supplements.

    Behavior's bound split audit is recomputed against the current frozen native
    geometry, including reverse traversals and every other supplement. A fresh
    self-declared group identifier cannot bypass physical split isolation.
    """
    root = Path(corpus).resolve()
    old, complete = discover_legacy(root)
    rows = [EpisodeRef('legacy', r['path'], r['sha256'], r['split'], r['group'],
        r['family'], r['labels']) for r in old]
    held = active_holds(root)
    for path in sorted((root/'accepted').glob('*/behavior-manifest.json')):
        if path.parent.name.casefold() in held:
            continue
        manifest = _read(path)
        if (manifest.get('status') != 'accepted' or manifest.get('training_ready') is not True
                or manifest.get('capture_scope') != 'production'):
            raise ValueError('nonadmitted behavior manifest in accepted corpus')
        saved_path = path.parent/'behavior-split.json'
        if manifest.get('files', {}).get(saved_path.name) != sha256_file(saved_path):
            raise ValueError('behavior split receipt is not hash bound')
        saved = _read(saved_path)
        task_path = path.parent/'candidate-task.json'
        task_sha = sha256_file(task_path)
        if manifest['files'].get(task_path.name) != task_sha:
            raise ValueError('behavior task is not hash bound')
        audit = refresh_behavior_split(root,_read(task_path),task_sha,saved,manifest.get('split'))
        assignments = {a['group_id']:a['split'] for a in audit['baseline_document']['assignments']}
        if any(assignments.get(r.group) != r.split for r in rows if r.kind == 'legacy'):
            raise ValueError('legacy index differs from behavior frozen split baseline')
        section = audit['candidate_section']
        rows.append(EpisodeRef('behavior', str(path.resolve()), sha256_file(path),
            manifest['split'], section['section_id'], section['family'], manifest['accepted_label_count']))
    if len({(r.kind, r.path.casefold()) for r in rows}) != len(rows):
        raise ValueError('duplicate mixed manifest identity')
    if any(not isinstance(r.labels, int) or r.labels <= 0 for r in rows):
        raise ValueError('mixed corpus contains an empty accepted episode')
    if verify_payloads:
        for row in rows:
            # Split audit includes test geometry; no test images or labels enter
            # the objective or model-selection pipeline.
            if row.split != 'test':
                with open_episode(row, root):
                    pass
    return rows, complete


def qualification(rows, recipe=None):
    recipe = recipe or Recipe()
    counts = {kind: {split: dict(episodes=0, labels=0) for split in ('train','validation','test')}
              for kind in KINDS}
    for row in rows:
        value = counts[row.kind][row.split]
        value['episodes'] += 1
        value['labels'] += row.labels
    missing = [f'{kind}:{split}' for kind in KINDS for split in ('train','validation')
               if not counts[kind][split]['episodes']]
    return dict(ready=not missing, missing_required_cohorts=missing, cohorts=counts,
        recipe=recipe.document(), recipe_sha256=fingerprint(recipe.document()),
        index_sha256=fingerprint([asdict(r) for r in sorted(rows, key=lambda r:r.key)]),
        optimizer_step_performed=False, production_activation_performed=False,
        test_split_used=False)


def _guard(episode, expected_sha=None, *, split=None):
    if isinstance(episode, AdmittedBehaviorSequence):
        episode.assert_admitted()
    elif isinstance(episode, SequenceEpisode):
        if episode.manifest.get('status') != 'accepted' or episode.manifest.get('admission', {}).get('passed') is not True:
            raise ValueError('legacy sequence requires accepted independent admission')
        assert_episode_not_held(episode.path)
        episode._verify_file_inventory(True)
    else:
        raise TypeError('only independently admitted legacy or behavior sequences are allowed')
    if split is not None and episode.manifest.get('split') != split:
        raise ValueError(f'only {split} split allowed for this operation')
    if expected_sha is not None and sha256_file(episode.path) != expected_sha:
        raise ValueError('accepted manifest changed during operation')


@contextmanager
def open_episode(row, corpus):
    if row.kind not in KINDS or row.split not in ('train','validation'):
        raise ValueError('only train/validation mixed sequences may be opened')
    path, root = Path(row.path).resolve(), Path(corpus).resolve()
    if path.parent.parent != root/'accepted':
        raise ValueError('mixed sequence must be directly inside corpus accepted')
    if sha256_file(path) != row.sha256:
        raise ValueError('accepted manifest changed before loading')
    episode = (AdmittedBehaviorSequence(path, corpus_root=root) if row.kind == 'behavior'
               else SequenceEpisode(path))
    try:
        _guard(episode, row.sha256, split=row.split)
        if len(episode.labels['label_indices']) != row.labels:
            raise ValueError('index and loaded accepted label counts differ')
        yield episode
        _guard(episode, row.sha256, split=row.split)
    finally:
        episode.close()


def gradient_counts(model):
    result = {}
    for role in ('vision','policy'):
        gradients = [p.grad for p in getattr(model, role).parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError(f'missing or nonfinite {role} gradients')
        result[role] = sum(int(g.count_nonzero()) for g in gradients)
        if not result[role]:
            raise ValueError(f'zero {role} gradients')
    return result


def backward_pair(legacy, behavior, student, reference, *, recipe=None, progress=None):
    """Exact weighted gradient of both episode objectives, without updating weights.

    Each existing backward implementation resets gradients. Retaining the first
    gradient tensors explicitly prevents the second pass from erasing it. Both
    passes see identical student weights and independent dense recurrent histories.
    """
    recipe = recipe or Recipe()
    if not isinstance(legacy, SequenceEpisode) or not isinstance(behavior, AdmittedBehaviorSequence):
        raise TypeError('mixed step requires one legacy and one admitted behavior sequence')
    manifests = [sha256_file(e.path) for e in (legacy, behavior)]
    before, baseline = parameter_digest(student), parameter_digest(reference)
    reports = {}
    saved = None
    student.zero_grad(set_to_none=True)
    try:
        for kind, episode, expected, function, weight in zip(KINDS, (legacy,behavior), manifests,
                (legacy_backward,behavior_backward), (recipe.legacy_weight,recipe.behavior_weight)):
            _guard(episode, expected, split='train')
            callback = None if progress is None else lambda stage,tick,total,k=kind: progress(k+':'+stage,tick,total)
            prediction, report = function(episode, student, reference, device='cpu', progress=callback)
            if (not bool(torch.isfinite(prediction).all()) or not math.isfinite(report['loss'])
                    or report['valid_labels'] <= 0):
                raise ValueError('nonfinite or empty mixed objective: '+kind)
            report = dict(report, nonzero_gradients=gradient_counts(student), weight=weight)
            reports[kind] = report
            _guard(episode, expected, split='train')
            if saved is None:
                # zero_grad(set_to_none=True) in the next backward detaches these
                # tensors from the parameters but does not invalidate their data.
                saved = {name:p.grad.mul_(weight) for name,p in student.named_parameters() if p.grad is not None}
            else:
                for name, parameter in student.named_parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(weight)
                        if name in saved:
                            parameter.grad.add_(saved[name])
                    elif name in saved:
                        parameter.grad = saved[name]
        for episode, expected in zip((legacy,behavior), manifests):
            _guard(episode, expected, split='train')
        if parameter_digest(student) != before or parameter_digest(reference) != baseline:
            raise ValueError('backward modified model parameters or frozen baseline')
        if any(p.grad is not None for p in reference.parameters()):
            raise ValueError('frozen baseline acquired gradients')
        gradients = gradient_counts(student)
        return dict(objective_version=OBJECTIVE_VERSION,recipe_sha256=fingerprint(recipe.document()),
            loss=sum(reports[k]['weight']*reports[k]['loss'] for k in KINDS),cohorts=reports,
            nonzero_gradients=gradients,optimizer_step_performed=False, test_split_used=False,
            student_parameter_sha256=before,baseline_parameter_sha256=baseline)
    except BaseException:
        student.zero_grad(set_to_none=True)
        raise


def behavior_lateral_score(report):
    return lateral_score(report)+LossConfig().clearance_weight*float(report.get('obstacle_loss',0.))


@torch.no_grad()
def evaluate_behavior(episode, student, reference, progress=None):
    expected = sha256_file(episode.path)
    _guard(episode, expected, split='validation')
    if any(p.requires_grad or p.grad is not None for p in reference.parameters()):
        raise ValueError('validation reference must be frozen without gradients')
    def evaluate(model, kind):
        ticks = 0
        def forward(inputs):
            nonlocal ticks
            output = model(inputs); ticks += 1
            if progress and ticks % 100 == 0: progress('behavior:validation_'+kind,ticks,len(episode))
            return output.detach()
        return student_rollout(episode,forward,device='cpu')[0]
    baseline, prediction = evaluate(reference,'reference'),evaluate(student,'student')
    if not bool(torch.isfinite(baseline).all() and torch.isfinite(prediction).all()):
        raise ValueError('nonfinite behavior validation output')
    results = []
    for index, label in enumerate(episode.behavior_labels):
        original = baseline[index:index+1]
        value, report = model_behavior_loss(prediction[index:index+1],original,[label.corridor],adapter=ROLLOUT_PLAN)
        base_value, base_report = model_behavior_loss(original,original,[label.corridor],adapter=ROLLOUT_PLAN)
        if report['valid_labels'] != base_report['valid_labels']:
            raise ValueError('behavior validation support changed with candidate')
        if not report['valid_labels']:
            continue
        results.append(dict(label_index=label.label_index,record_sha256=label.record_sha256,
            loss=behavior_lateral_score(report),baseline_loss=behavior_lateral_score(base_report),
            optimization_loss=float(value),baseline_optimization_loss=float(base_value),
            progress_guard_loss=report['progress_loss'],baseline_progress_guard_loss=base_report['progress_loss']))
    if not results:raise ValueError('validation has no supported behavior labels')
    _guard(episode,expected,split='validation')
    fields = ('loss','baseline_loss','optimization_loss','baseline_optimization_loss',
              'progress_guard_loss','baseline_progress_guard_loss')
    return dict(labels=len(results),**{key:sum(r[key] for r in results)/len(results) for key in fields},
        label_scores=results,selection_metric=SELECTION_METRIC)


def evaluate(rows, corpus, student, reference, *, recipe=None, progress=None):
    recipe = recipe or Recipe()
    selected = sorted((r for r in rows if r.split == 'validation'),key=lambda r:r.key)
    if {r.kind for r in selected} != set(KINDS):
        raise ValueError('both legacy and behavior heldout cohorts are required')
    before, baseline = parameter_digest(student), parameter_digest(reference)
    results = []
    for row in selected:
        with open_episode(row, corpus) as episode:
            result = (evaluate_episode(episode,student,reference,progress) if row.kind == 'legacy'
                      else evaluate_behavior(episode,student,reference,progress))
        results.append(dict(**result,kind=row.kind,episode=row.path,manifest_sha256=row.sha256))
    if parameter_digest(student) != before or parameter_digest(reference) != baseline:
        raise ValueError('heldout evaluation changed model weights')
    fields = ('loss','baseline_loss','optimization_loss','baseline_optimization_loss',
              'progress_guard_loss','baseline_progress_guard_loss')
    cohorts = {}
    for kind in KINDS:
        group = [r for r in results if r['kind'] == kind]
        count = sum(r['labels'] for r in group)
        cohorts[kind] = dict(labels=count,episodes=len(group),
            cohort=fingerprint(sorted(r['manifest_sha256'] for r in group)),
            **{k:sum(r[k]*r['labels'] for r in group)/count for k in fields})
    weights = dict(legacy=recipe.legacy_weight,behavior=recipe.behavior_weight)
    return dict(selection_metric=SELECTION_METRIC,objective_version=OBJECTIVE_VERSION,
        recipe_sha256=fingerprint(recipe.document()),cohort=fingerprint([r.sha256 for r in selected]),
        labels=sum(r['labels'] for r in results),episodes=len(results),cohorts=cohorts,rows=results,
        **{k:sum(weights[kind]*cohorts[kind][k] for kind in KINDS) for k in fields},test_split_used=False)


def can_promote_mixed(candidate, incumbent=None):
    """No trade of old lane quality for behavior quality, or vice versa."""
    if candidate.get('selection_metric') != SELECTION_METRIC:
        raise ValueError('mixed promotion requires its explicit selection metric')
    if set(candidate.get('cohorts',{})) != set(KINDS):
        raise ValueError('mixed promotion requires both heldout cohorts')
    if incumbent is not None and candidate.get('recipe_sha256') != incumbent.get('recipe_sha256'):
        raise ValueError('cannot compare different mixed objective recipes')
    for kind in KINDS:
        cohort = candidate['cohorts'][kind]
        if not all(math.isfinite(cohort[k]) for k in ('loss','baseline_loss')) or cohort['labels'] <= 0:
            raise ValueError('invalid heldout cohort score')
        if cohort['loss'] > cohort['baseline_loss']:
            return False
        if incumbent is not None:
            old = incumbent['cohorts'][kind]
            if cohort['cohort'] != old['cohort'] or cohort['labels'] != old['labels']:
                raise ValueError('incumbent requires the exact same heldout cohort/support')
            if cohort['loss'] > old['loss']:
                return False
    return can_promote(candidate,incumbent)
